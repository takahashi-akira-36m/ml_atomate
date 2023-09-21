"""
This code is created based on PHYSBO's policy.py with some modifications.
"""

import copy
from typing import Tuple, Optional, List
from logging import getLogger, INFO
import datetime

import numpy as np
import scipy
from physbo import variable
from physbo.gp import predictor as gp_predictor
from physbo.blm import predictor as blm_predictor
from physbo.search import utility
from physbo.search.discrete_multi.policy import policy, history, _run_simulator
from physbo.misc import set_config

logger = getLogger(__name__)
logger.setLevel(INFO)


def _get_fmean_fstd(predictor_list, training_list, test_list):
    fmean = [
        predictor.get_post_fmean(training, test)
        for predictor, training, test in zip(predictor_list, training_list, test_list)
    ]
    fcov = [
        predictor.get_post_fcov(training, test)
        for predictor, training, test in zip(predictor_list, training_list, test_list)
    ]

    # shape: (N, n_obj)
    fmean = np.array(fmean).T
    fstd = np.sqrt(np.array(fcov)).T
    return fmean, fstd


def range_acquisition_function_multi(predictor_list,
                                     training_list,
                                     test_list: List[variable],
                                     limit: Tuple[Tuple[Optional[float], Optional[float]]],
                                     ):
    fmean, fstd = _get_fmean_fstd(predictor_list, training_list, test_list)
    fmean_train, fstd_train = _get_fmean_fstd(predictor_list, training_list, training_list)

    # logger.debug(f"fmean, fstd, limit = {fmean}, {fstd}, {limit}")
    n_data = len(fmean)
    n_objectives = len(predictor_list)
    scores = np.zeros(n_data)
    for n in range(n_data):
        score = 1
        for o in range(n_objectives):
            logger.debug(f"*******training_list[{o}] (n = {n})********")
            logger.debug("X")
            logger.debug(training_list[o].X)
            logger.debug("Z")
            logger.debug(training_list[o].Z)
            logger.debug("t")
            logger.debug(training_list[o].t)
            logger.debug("*******fmean*********")
            logger.debug(fmean)
            logger.debug("*******fstd*********")
            logger.debug(fstd)
            limit_low, limit_high = limit[o]
            fm = fmean[n][o]
            fs = fstd[n][o]
            temp_low = scipy.stats.norm.cdf((limit_low - fm) / fs) if not np.isnan(limit_low) else 0
            temp_high = scipy.stats.norm.cdf((limit_high - fm) / fs) if not np.isnan(limit_high) else 1
            score = score * (temp_high - temp_low)
            logger.debug(f"test_list: {test_list[o].X}")
            # logger.debug(f"test_list: {test[o].t[n]}")
            logger.debug(f"fmean = {fmean[n][o]}, fstd = {fstd[n][o]}")
            logger.debug(f"n = {n}, o = {o}, limit={limit_low:3f}, {limit_high:3f} "
                         f"temp_high = {temp_high:3f}, temp_low = {temp_low:3f}, "
                         f"diff={temp_high-temp_low:3f} score = {score:3f}")
        scores[n] = score
    return scores


def my_score(mode, predictor_list, test_list,
             training_list: Optional[variable] = None,
             **kwargs):
    """
    Calculate scores (acquisition function) for test data.

    Parameters
    ----------
    mode: str
        Kind of score.

        "EI", "PI", and "TS" are available.

    predictor_list: predictor object
        Base class is defined in physbo.predictor.

    training_list: physbo.variable
        Training dataset.
        If the predictor is not trained, use this for training.

    test_list: list of physbo.variable
        Inputs

    Other Parameters
    ----------------
    fmax: float
        Max value of mean of posterior probability distribution.
        If not set, the maximum value of posterior mean for training is used.
        Used only for mode == "EI" and "PI"

    alpha: float
        noise for sampling source (default: 1.0)
        Used only for mode == "TS"

    limit: Tuple[float, float]


    Returns
    -------
    score: numpy.ndarray

    Raises
    ------
    NotImplementedError
        If unknown mode is given
    """

    if test_list[0].X.shape[0] == 0:
        return np.zeros(0)

    if mode == "RANGE":
        limit = kwargs["limit"]
        return range_acquisition_function_multi(predictor_list,
                                                training_list,
                                                test_list,
                                                limit=limit)
    else:
        raise NotImplementedError(f"ERROR: mode must be EI, PI, TS, or RANGE. (actual: {mode})")


class Policy(policy):

    new_data_list: List[Optional[variable]]

    def __init__(
            self, test_X: List, num_objectives, comm=None, config=None, initial_data=None, log_dir=None
    ):
        print(f"config = {config}")
        self._log_dir = log_dir
        self.num_objectives = num_objectives
        self.history = history(num_objectives=self.num_objectives)

        self.training_list = [variable() for _ in range(self.num_objectives)]
        self.predictor_list = [None for _ in range(self.num_objectives)]
        self.test_list = [
            self._make_variable_X(test_X[i]) for i in range(self.num_objectives)
        ]
        self.new_data_list = [None for _ in range(self.num_objectives)]

        self.actions = np.arange(0, np.array(test_X).shape[1])
        if config is None:
            self.config = set_config()
        else:
            self.config = config

        self.TS_candidate_num = None

        if initial_data is not None:
            if len(initial_data) != 2:
                msg = "ERROR: initial_data should be 2-elements tuple or list (actions and objectives)"
                raise RuntimeError(msg)
            actions, fs = initial_data
            if fs.shape[1] != self.num_objectives:
                msg = "ERROR: initial_data[1].shape[1] != num_objectives"
                raise RuntimeError(msg)
            if len(actions) != fs.shape[0]:
                msg = "ERROR: len(initial_data[0]) != initial_data[1].shape[0]"
                raise RuntimeError(msg)
            self.write(actions, fs)
            self.actions = sorted(list(set(self.actions) - set(actions)))

        if comm is None:
            self.mpicomm = None
            self.mpisize = 1
            self.mpirank = 0
        else:
            self.mpicomm = comm
            self.mpisize = comm.size
            self.mpirank = comm.rank
            self.actions = np.array_split(self.actions, self.mpisize)[self.mpirank]
        print(f"rank = {self.mpirank}, self.actions = {self.actions}")

    def _model(self, i):
        training = self.training_list[i]
        predictor = self.predictor_list[i]
        test = self.test_list[i]
        new_data = self.new_data_list[i]
        return {"training": training, "predictor": predictor, "test": test, "new_data": new_data}

    def _learn_hyperparameter(self, num_rand_basis):
        for i in range(self.num_objectives):
            m = copy.deepcopy(self._model(i))
            predictor = m["predictor"]
            training = m["training"]
            test = m["test"]
            predictor.fit(training, num_rand_basis)
            test.Z = predictor.get_basis(test.X)
            training.Z = predictor.get_basis(training.X)
            predictor.prepare(training)

            # DEBUG
            self.predictor_list[i] = predictor
            self.test_list[i] = test
            self.training_list[i] = training
            # END DEBUG

            self.new_data_list[i] = None
            # self.predictor_list[i].fit(self.training_list[i], num_rand_basis)
            # self.test_list[i].Z = self.predictor_list[i].get_basis(self.test_list[i].X)
            # self.training_list[i].Z = self.predictor_list[i].get_basis(self.training_list[i].X)
            # self.predictor_list[i].prepare(self.training_list[i])
            # self.new_data_list[i] = None

    def _update_predictor(self):
        for i in range(self.num_objectives):
            if self.new_data_list[i] is not None:
                self.predictor_list[i].update(self.training_list[i], self.new_data_list[i])
                self.new_data_list[i] = None

    def bayes_search(
            self,
            limit,
            training_list=None,
            max_num_probes=None,
            num_search_each_probe=1,
            predictor_list=None,
            is_disp=True,
            disp_pareto_set=False,
            simulator=None,
            score="RANGE",
            interval=0,
            num_rand_basis=0,
    ):

        if self.mpirank != 0:
            is_disp = False

        if max_num_probes is None:
            max_num_probes = 1
            simulator = None

        logger.info(f"{datetime.datetime.now()}: learning gp start")
        is_rand_expans = False if num_rand_basis == 0 else True

        if training_list is not None:
            self.training_list = training_list

        if predictor_list is None:
            if is_rand_expans:
                self.predictor_list = [
                    blm_predictor(self.config) for _ in range(self.num_objectives)
                ]
            else:
                self.predictor_list = [
                    gp_predictor(self.config) for _ in range(self.num_objectives)
                ]
        else:
            self.predictor_list = predictor_list

        if max_num_probes == 0 and interval >= 0:
            self._learn_hyperparameter(num_rand_basis)

        N = int(num_search_each_probe)
        logger.info(f"{datetime.datetime.now()}: learning gp end")

        for n in range(max_num_probes):
            logger.info(f"{datetime.datetime.now()}: {n}-th probe")

            if utility.is_learning(n, interval):
                self._learn_hyperparameter(num_rand_basis)
            else:
                self._update_predictor()

            # if num_search_each_probe != 1:
            #     utility.show_start_message_multi_search_mo(self.history.num_runs, score)

            K = self.config.search.multi_probe_num_sampling
            action = self._get_actions(score, N, K, limit)

            if len(action) == 0:
                if self.mpirank == 0:
                    print("WARNING: All actions have already searched.")
                return copy.deepcopy(self.history)

            if simulator is None:
                return action

            t = _run_simulator(simulator, action, self.mpicomm)
            self.write(action, t)

            if is_disp:
                utility.show_search_results_mo(
                    self.history, N, disp_pareto_set=disp_pareto_set
                )
        self._update_predictor()
        return copy.deepcopy(self.history)

    def write(self, action, t, arg_x=None):
        self.history.write(t, action)
        action = np.array(action)
        t = np.array(t)

        for obj in range(self.num_objectives):
            test = self.test_list[obj]
            predictor = self.predictor_list[obj]

            if arg_x is None:
                X = test.X[action, :]
                Z = test.Z[action, :] if test.Z is not None else None
            else:
                raise ValueError(f"Unexpected situation. Check this implement (len(Z) really 1?)")
                X = copy.deepcopy(arg_x)
                Z = predictor.get_basis(X) if predictor is not None else None

            if self.new_data_list[obj] is None:
                self.new_data_list[obj] = variable(X, t[:, obj], Z)
            else:
                self.new_data_list[obj].add(X=X, t=t[:, obj], Z=Z)
            self.training_list[obj].add(X=X, t=t[:, obj], Z=Z)

    def _get_marginal_score(self, mode, chosen_actions, K, limit):
        """
        Getting marginal scores.

        Parameters
        ----------
        mode: str
            The type of aquision funciton.
            TS (Thompson Sampling), EI (Expected Improvement) and PI (Probability of Improvement) are available.
            These functions are defined in score.py.
        chosen_actions: numpy.ndarray
            Array of selected actions.
        K: int
            The total number of search candidates.
        alpha: float
            not used.

        Returns
        -------
        f: list
            N dimensional scores (score is defined in each mode)
        """
        f = np.zeros((K, len(self.actions)), dtype=float)
        new_test_list = []
        virtual_t_list = []
        for obj in range(self.num_objectives):
            new_test_local = self.test_list[obj].get_subset(chosen_actions)
            if self.mpisize == 1:
                new_test = new_test_local
            else:
                new_test = variable()
                for nt in self.mpicomm.allgather(new_test_local):
                    new_test.add(X=nt.X, t=nt.t, Z=nt.Z)

            virtual_t = self.predictor_list[obj].get_predict_samples(self.training_list[obj], new_test, K)

            new_test_list.append(new_test)
            virtual_t_list.append(virtual_t)

        for k in range(K):
            # logger.info(f"k = {k}, core = {psutil.Process().cpu_num()}, {datetime.datetime.now()}")
            predictor_list = []
            training_list = []
            for obj in range(self.num_objectives):
                predictor = copy.deepcopy(self.predictor_list[obj])
                train = copy.deepcopy(self.training_list[obj])
                virtual_train = new_test_list[obj]
                virtual_train.t = virtual_t_list[obj][k, :]

                if virtual_train.Z is None:
                    train.add(virtual_train.X, virtual_train.t)
                else:
                    train.add(virtual_train.X, virtual_train.t, virtual_train.Z)

                predictor.update(train, virtual_train)
                predictor_list.append(predictor)
                training_list.append(train)
            f[k, :] = self.get_score(mode,
                                     predictor_list=predictor_list,
                                     training_list=training_list,
                                     parallel=False,
                                     limit=limit)
        return np.mean(f, axis=0)

    def _get_actions(self, mode, N, K, limit):
        f = self.get_score(mode=mode, limit=limit, parallel=False)
        champion, local_champion, local_index = self._find_champion(f)
        if champion == -1:
            return np.zeros(0, dtype=int)
        if champion == local_champion:
            self.actions = self._delete_actions(local_index)

        chosen_actions = [champion]
        logger.info(f"{datetime.datetime.now()}: marginal score start")
        for n in range(1, N):
            logger.info(f"{datetime.datetime.now()}: n = {n}")
            f = self._get_marginal_score(mode, chosen_actions[0:n], K, limit=limit)
            logger.info(f"n = {n}, marginal_score f = {f}")
            # f = self.get_score(mode=mode, limit=limit, parallel=False)
            champion, local_champion, local_index = self._find_champion(f)
            if champion == -1:
                break
            chosen_actions.append(champion)
            if champion == local_champion:
                self.actions = self._delete_actions(local_index)
        logger.info(f"{datetime.datetime.now()}: marginal score start")
        return np.array(chosen_actions)

    def get_score(
            self,
            mode,
            actions=None,
            xs=None,
            predictor_list=None,
            training_list=None,
            pareto=None,
            parallel=True,
            limit=None,
    ):
        if predictor_list is None:
            predictor_list = self.predictor_list
        if training_list is None:
            training_list = self.training_list
        if pareto is None:
            pareto = self.history.pareto

        if training_list[0].X is None or training_list[0].X.shape[0] == 0:
            msg = "ERROR: No training data is registered."
            raise RuntimeError(msg)

        if predictor_list == [None] * self.num_objectives:
            self._warn_no_predictor("get_score()")
            for i in range(self.num_objectives):
                predictor_list[i] = gp_predictor(self.config)
                predictor_list[i].fit(training_list[i], 0)
                predictor_list[i].prepare(training_list[i])

        if xs is not None:
            if actions is not None:
                raise RuntimeError("ERROR: both actions and xs are given")
            if isinstance(xs, variable):
                test = xs
            else:
                test = variable(X=xs)
            if parallel and self.mpisize > 1:
                actions = np.array_split(np.arange(test.X.shape[0]), self.mpisize)
                test = test.get_subset(actions[self.mpirank])
        else:
            if actions is None:
                actions = self.actions
            else:
                if isinstance(actions, int):
                    actions = [actions]
                if parallel and self.mpisize > 1:
                    actions = np.array_split(actions, self.mpisize)[self.mpirank]
            # test = self.test_list[0].get_subset(actions)
            test_list = [self.test_list[o].get_subset(actions) for o in range(self.num_objectives)]

        f = my_score(
            mode,
            predictor_list=predictor_list,
            training_list=training_list,
            test_list=test_list,
            pareto=pareto,
            reduced_candidate_num=self.TS_candidate_num,
            limit=limit,
            log_dir=self._log_dir,
        )
        if parallel and self.mpisize>1:
            fs = self.mpicomm.allgather(f)
            f = np.hstack(fs)
        return f
