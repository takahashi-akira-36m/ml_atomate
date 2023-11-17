from ml_atomate.data_manager import DataManager
from math import log10
from copy import deepcopy
from collections import defaultdict
import abc
import os
from logging import getLogger, DEBUG, basicConfig
from typing import Dict, List, Union, Optional, Iterable, Set, Callable, Sequence, Tuple
from itertools import chain
import numpy as np
import pandas as pd
from enum import Enum
from datetime import datetime
from ml_atomate.physbo_customized.policy_ptr import policy_ptr
from ml_atomate.blox_kterayama.curiosity_sampling import stein_novelty
from physbo.search.discrete.policy import policy as physbo_discrete_policy
from physbo.search.discrete_multi.policy import policy as physbo_discrete_multi_policy
from pymongo.database import Database
import multiprocessing
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


logger = getLogger(__name__)
#  Implement keep running fw_id


class Conversion(Enum):
    no_conversion = "no_conversion"
    log = "log"

    def convert(self, x: Optional[float]) -> Optional[float]:
        # Use monotonically increasing function
        if x is None:
            return None
        if self == Conversion.no_conversion:
            return x
        elif self == Conversion.log:
            return log10(x)


class MlProcedure:

    def __init__(self,
                 descriptor_data_frame: pd.DataFrame,
                 objective_data_frame: pd.DataFrame,
                 searched_index: pd.Index,
                 candidate_index: pd.Index,
                 descriptor_col: Sequence[str],
                 objective_col: Sequence[str],
                 db: Database,
                 dropped_descriptors: pd.Index,
                 transformer: type[TransformerMixin] = StandardScaler,
                 conversion: Optional[Sequence[str | Conversion] | Dict[str, str | Conversion]] = None
                 ):
        self._searched_index = searched_index
        self._candidate_index = candidate_index
        self._descriptor_col = list(descriptor_col)
        self._objective_col = list(objective_col)
        self._scaler: Dict[str, TransformerMixin] = defaultdict(transformer)
        self._descriptor_data_frame = descriptor_data_frame
        self._objective_data_frame = objective_data_frame
        if conversion is None:
            self._conversion = {o: Conversion.no_conversion for o in self.objective_col}
        elif isinstance(conversion, Sequence):
            self._conversion = {o: Conversion(c) for o, c in zip(self.objective_col, conversion)}
        elif isinstance(conversion, dict):
            self._conversion = {k: Conversion(v) for k, v in conversion.items()}
        else:
            raise ValueError(f"Invalid type conversion {conversion}")
        for d in self._descriptor_col:
            self._descriptor_data_frame[d] = self._scaler[d].fit_transform(self._descriptor_data_frame[[d]].values)
        for o in self._objective_col:
            self._objective_data_frame[o] = self._objective_data_frame[o].apply(self._conversion[o].convert)
            self._objective_data_frame[o] = self._scaler[o].fit_transform(self._objective_data_frame[[o]].values)
        self._dropped_descriptors = dropped_descriptors
        self._db = db
        self._selected_descriptors = None
        self._prune_details = None

    @classmethod
    def from_data_manager(cls, data_manager: DataManager, descriptor_dataframe: pd.DataFrame,
                          transformer: type[TransformerMixin] = StandardScaler,
                          conversion: Optional[Sequence[Conversion] | Dict[str, Conversion]] = None):
        descriptor_dataframe = descriptor_dataframe.copy()
        if data_manager.index_name is not None:
            descriptor_dataframe.set_index(data_manager.index_name, inplace=True)
        descriptor_dataframe = descriptor_dataframe.loc[data_manager.not_fizzled_index]
        descriptors_to_delete = descriptor_dataframe.isnull().any(axis=0)
        descriptor_dataframe.drop(columns=descriptors_to_delete[descriptors_to_delete == True].index, inplace=True)
        dropped_descriptors: pd.Index = descriptors_to_delete[descriptors_to_delete == True].index
        logger.info(f"These {len(dropped_descriptors)} descriptors are dropped, "
                    f"since at least one material doesn't have descriptor values. "
                    f"(remaining: {len(descriptor_dataframe.columns)}\nDropped descriptor: {dropped_descriptors}")
        return cls(descriptor_dataframe,
                   data_manager.not_fizzled_df.drop(columns="state"),
                   data_manager.searched_index,
                   data_manager.candidate_index,
                   descriptor_dataframe.columns,
                   data_manager.objective_paths,
                   data_manager.db,
                   dropped_descriptors,
                   transformer,
                   conversion
                   )

    def transform_param(self, val, label):
        if np.isnan(val) or val is None:
            return val
        if label in self.objective_col:
            val = self.conversion[label].convert(val)
        return self.scaler[label].transform(np.array([[val]]))[0][0]

    @property
    def conversion(self) -> Dict[str, Conversion]:
        return self._conversion

    @property
    def searched_index(self):
        return self._searched_index

    @property
    def candidate_index(self):
        return self._candidate_index

    @property
    def descriptor_col(self):
        return self._descriptor_col

    @property
    def objective_col(self):
        return self._objective_col

    @property
    def selected_descriptors(self) -> Optional[Dict[str, List[str]]]:
        return self._selected_descriptors

    @property
    def train_x(self) -> Dict[str, pd.DataFrame]:
        if self._selected_descriptors is None:
            return {k: self._descriptor_data_frame.loc[self.searched_index]
                    for k in self.objective_col}
        else:
            return {k: self._descriptor_data_frame[v].loc[self.searched_index]
                    for k, v in self._selected_descriptors.items()}

    @property
    def train_y(self) -> pd.DataFrame:
        return self._objective_data_frame.loc[self.searched_index]

    @property
    def candidate_x(self):
        if self._selected_descriptors is None:
            return {k: self._descriptor_data_frame.loc[self.candidate_index]
                    for k in self.objective_col}
        else:
            return {k: self._descriptor_data_frame[v].loc[self.candidate_index]
                    for k, v in self._selected_descriptors.items()}

    @property
    def scaler(self) -> Dict[str, TransformerMixin]:
        return self._scaler

    def get_priority(self,
                     model_type: type[BaseEstimator | physbo_discrete_policy],
                     acquisition_function: Optional[Callable[[Iterable[float]], float]],
                     model_params: Optional[dict] = None,
                     acquisition_params: Optional[dict] = None,
                     num_probes: Optional[int] = None
                     ) -> List[str]:
        if model_params is None:
            model_params = dict()
        if acquisition_params is None:
            acquisition_params = dict()
        if num_probes is None:
            num_probes = len(self.candidate_index)
        if issubclass(model_type, physbo_discrete_policy):  # Discrete, multi_discrete
            # Type of policy
            num_objectives = len(self.objective_col)
            all_x = [pd.concat([self.train_x[k], self.candidate_x[k]]).values for k in self.train_x.keys()]
            train_index = list(range(len(self.searched_index)))
            train_y = self.train_y.values
            index_comp_map = {i: idx for i, idx in enumerate(list(self.searched_index) + list(self.candidate_index))}
            comm = model_params.get("comm", acquisition_params.get("comm"))
            seed = model_params.pop("seed")
            if issubclass(model_type, policy_ptr):
                policy = model_type(all_x,
                                    num_objectives,
                                    initial_data=(train_index, train_y),
                                    comm=comm)
                if seed is not None:
                    policy.set_seed(seed)
                actions = policy.bayes_search(num_search_each_probe=num_probes,
                                              score=acquisition_params["score"],
                                              limit=acquisition_params["limit"],
                                              interval=0,
                                              **model_params)
            elif issubclass(model_type, physbo_discrete_multi_policy):
                if self.selected_descriptors is not None:
                    raise ValueError(f"PHYSBO policy_discrete_multi can't use pruned descriptor")
                policy = model_type(all_x[0],
                                    num_objectives,
                                    initial_data=(train_index, train_y),
                                    comm=comm)
                if seed is not None:
                    policy.set_seed(seed)
                actions = policy.bayes_search(num_search_each_probe=num_probes,
                                              score=acquisition_params["score"],
                                              interval=0,
                                              **model_params)
            else:
                if num_objectives >= 2:
                    raise ValueError(f"Please use policy_discrete_multi if you use {num_objectives} (>=2) objectives.")
                if self.selected_descriptors is not None:
                    raise ValueError(f"PHYSBO policy_discrete can't use pruned descriptor")
                policy = model_type(all_x[0],
                                    initial_data=(train_index, train_y[:, 0]),
                                    comm=comm)
                if seed is not None:
                    policy.set_seed(seed)
                actions = policy.bayes_search(num_search_each_probe=num_probes,
                                              score=acquisition_params["score"],
                                              interval=0,
                                              **model_params
                                              )

            actions = [index_comp_map[i] for i in actions]

        # sklearn
        elif issubclass(model_type, BaseEstimator):
            predictions = dict()
            for obj in self.objective_col:
                model = model_type(**model_params)
                model.fit(self.train_x[obj].values, self.train_y[obj].values)
                predictions[obj] = model.predict(self.candidate_x[obj].values)
            scores = \
                np.array(
                    [acquisition_function([predictions[obj][i] for obj in self.objective_col], **acquisition_params)
                     for i, _ in enumerate(self.candidate_index)])
            actions = [self.candidate_index[i] for i in scores.argsort()[::-1][:num_probes]]
        else:
            raise ValueError("Unknown model")
        return actions

    # BLOX requires standardized y_values
    def stein_novelty(self, y, sigma):
        return stein_novelty(y, self.train_y.values, sigma)

    # TODO: implement RFE, RFECV
    def prune_descriptors_rf_importance(self, n_descriptor_bayes, n_estimators, uses_permutation_importances):
        if self.selected_descriptors is not None:
            logger.warning("Descriptors are already pruned. Now pruning will be done again.")
        selected_descriptors = dict()  # {"bandgap": ["x0", "x2",...]}
        prune_details = dict()
        prune_details["method"] = "Random Forest Importance"
        prune_details["importances"] = dict()
        if uses_permutation_importances:
            prune_details["importances_each"] = list()
            prune_details["importances_std"] = list()
        prune_details["rfr_start"] = dict()
        prune_details["rfr_end"] = dict()
        #
        for obj in self.objective_col:
            logger.info(f"obj = {obj}, random forest...")

            # Train the model using the core sets
            n_cpus = multiprocessing.cpu_count()
            logger.info(f"Server info: {os.uname()}")
            logger.info(f'Number of CPUs: {n_cpus}')

            rfr = RandomForestRegressor(n_estimators=n_estimators,
                                        n_jobs=n_cpus,
                                        random_state=1)  # debug

            rfr_start = datetime.now()
            logger.info(f"random forest fit... : {rfr_start}")
            rfr.fit(self.train_x[obj].values, self.train_y[obj].values)
            rfr_end = datetime.now()
            logger.info(f"random forest fit ends : {rfr_end} ({rfr_end - rfr_start} [sec])")
            if uses_permutation_importances:
                permutation_importance_result = permutation_importance(rfr, self.train_x, self.train_y[obj], n_jobs=-1)
                importances = permutation_importance_result["importances_mean"]
                prune_details["importances_each"].append(permutation_importance_result["importances"])
                prune_details["importances_std"].append(permutation_importance_result["importances_std"])
            else:
                importances = rfr.feature_importances_
            indices = np.argsort(importances)[::-1]
            for f in range(self.train_x[obj].shape[1]):
                logger.info("%d. feature %s (%f)" % (f + 1, self.train_x[obj].columns[indices[f]], importances[indices[f]]))
            selected_descriptors[obj] = [self.train_x[obj].columns[indices[i]] for i in range(n_descriptor_bayes)]
            prune_details["importances"][obj] = importances
            prune_details["rfr_start"][obj] = rfr_start
            prune_details["rfr_end"][obj] = rfr_end
            self._prune_details = prune_details
        self._selected_descriptors = selected_descriptors
