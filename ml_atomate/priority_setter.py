import argparse
import sys
import copy
from collections import defaultdict
from datetime import datetime
from itertools import count, chain
from typing import Dict, List
from pathlib import Path
from atomate.utils.utils import get_database
import random
import json
import os
import time
import traceback
from math import log10
from logging import getLogger, DEBUG, basicConfig
import numpy as np
import pandas as pd
import multiprocessing
from pymongo import database
from fireworks import LaunchPad
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor

from ml_atomate.physbo_customized.policy_ptr import Policy
from ml_atomate.utils.util import get_from_mongo_like_str, parse_objective
from ml_atomate.blox_kterayama.curiosity_sampling import stein_novelty
import importlib


logger = getLogger(__name__)
now = datetime.now()
FW_LOGGING_FORMAT = '%(asctime)s %(levelname)s %(message)s'  # format for loggers
basicConfig(filename=f"priority_{now.year}{now.month}{now.day}_{now.hour}{now.minute}{now.second}{now.microsecond}.log",
            format=FW_LOGGING_FORMAT)
logger.setLevel(DEBUG)

DEFAULT_PRIORITY = -1


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--db_file", "-df",
                        help="path to db.json",
                        type=str,
                        required=True)

    parser.add_argument("--builder", "-bld",
                        help="Specify python file containing run_builder function",
                        type=str)

    parser.add_argument("--descriptor_csv", "-dc",
                        help="path to descriptor.csv",
                        type=str,
                        required=True)

    parser.add_argument("--objective", "-o",
                        help="Set prediction objective. "
                             "When you use PTR function,"
                             "you can use mongolike specification (e.g. other_prop.energy_per_atom)"
                             "Write range together, like "
                             "bandstructure_hse.bandgap 4.0, dielectric.epsilon_avg 30.0,",
                        type=str,
                        nargs="+",
                        required=True
                        )

    parser.add_argument("--conversion", "-c",
                        help="Select no_conversion or log",
                        type=str,
                        nargs="+",
                        )

    parser.add_argument("--property_descriptor", "-pd",
                        help="Set property when you want to use other property as descriptor, e.g. GGA band_gap.",
                        type=str,
                        nargs="+",
                        )

    parser.add_argument("--n_estimators", "-e",
                        help="The number of trees "
                             "for the Random Forest Regression",
                        type=int,
                        default=1000)

    parser.add_argument("--n_seeds", "-ns",
                        help="The number of seeds for CV",
                        type=int,
                        default=1)

    parser.add_argument("--n_cv_folds", "-n",
                        help="The N value for N-folds cross validation",
                        type=int,
                        default=0)

    parser.add_argument("--permutation_importance", "-pi",
                        help="Use permutation importance to prune descriptors",
                        action="store_true")

    # For black_box optimization
    parser.add_argument("--random_seed", "-rs",
                        help="Random seed for bayes",
                        type=int,
                        default=0)

    parser.add_argument("--all_descriptor", "-ad",
                        help="All descriptor is used",
                        action="store_true")

    parser.add_argument("--n_descriptor", "-nd",
                        help="Number of descriptors using Gaussian process (pruned by random forest)",
                        type=int,
                        default=10)

    parser.add_argument("--n_write_server", "-nws",
                        help="Number of materials to set priority in mongodb.",
                        type=int,
                        default=-1)

    parser.add_argument("--n_rand_basis", "-nrb",
                        help="Number of basis. (See PHYSBO code.)",
                        type=int,
                        default=0)

    parser.add_argument("--monitor", "-m",
                        help="Monitoring time (sec)",
                        type=int,
                        default=60)

    parser.add_argument("--restart_dir", "-rd",
                        help="Restart by using {restart_dir}/step_XX/result.json. Also can be specify step_XX dir.",
                        type=str)

    parser.add_argument("--blox", "-b",
                        help="Use blox",
                        action="store_true")

    parser.add_argument("--initial_priority", "-ip",
                        help="Specify step_*/result.json. Same initial priority will be set",
                        type=str)

    args = parser.parse_args()
    return args


def get_comp_fwids_map(db: database) -> Dict[str, List[int]]:
    logger.info("Get map of fw_id and composition...")
    fw_col = db["fireworks"]
    dd = defaultdict(list)
    for doc in fw_col.find({}, ["name", "fw_id"]):
        # name: {compostiion}-{task_label}
        comp = doc["name"].split("-")[0]
        dd[comp].append(doc["fw_id"])
    return dd


def comps_to_fwids(comp_array: List[str], comp_fwid_map: Dict[str, List[int]]):
    return list(chain.from_iterable(comp_fwid_map[comp] for comp in comp_array))


def fwids_to_comps(fw_id_array: List[int], comp_fwid_map: Dict[str, List[int]]):
    return [comp for comp, fw_ids in comp_fwid_map.items() if set(fw_id_array) & set(fw_ids)]


def val_in_limit(val, lim):
    flag = True
    if lim[0] is not None:
        flag = flag and (lim[0] <= val)
    if lim[1] is not None:
        flag = flag and (val <= lim[1])
    return flag


def get_fizzled_composition(known_compositions: List[str],
                            db: database,
                            comp_fwid_map: Dict[str, List[int]]):
    fizzled_fw_ids = []
    known_fw_ids = comps_to_fwids(known_compositions, comp_fwid_map)
    for fizzled_doc in db["fireworks"].find({"state": "FIZZLED", "fw_id": {"$nin": known_fw_ids}}, ["fw_id"]):
        fizzled_fw_ids.append(fizzled_doc["fw_id"])
    fizzled_composition = []
    for comp, fw_ids in comp_fwid_map.items():
        if set(fizzled_fw_ids) & set(fw_ids):
            fizzled_composition.append(comp)
    return list(fizzled_composition)


def convert(value: float, conv_type: str):
    # Use monotonically increasing function
    if conv_type == "no_conversion":
        return value
    elif conv_type == "log":
        return log10(value)
    else:
        raise ValueError(f"Invalid conversion type {conv_type}")


def get_materials_properties(db: database,
                             property_paths: List[str],
                             known_compositions: List[str],
                             conversion: List[str],):
    output_dict = dict()
    query = {p: {"$exists": True} for p in property_paths}
    query["formula_pretty"] = {"$nin": known_compositions}
    target_compositions = \
        list(doc["formula_pretty"] for doc in db["materials"].find(query, {"formula_pretty": True}))
    for composition in target_compositions:
        projection = {".".join([k for k in p.split(".") if not k.isdecimal()]): True for p in property_paths}
        query = {"formula_pretty": composition}
        doc = db["materials"].find_one(query, projection)
        if doc is not None:
            try:
                output_dict[composition] = [convert(get_from_mongo_like_str(doc, p), conv_type)
                                            for p, conv_type in zip(property_paths, conversion)]
            except:
                logger.exception("<---")
                logger.exception(f"There was an error processing composition: {composition}")
                logger.exception(traceback.format_exc())
                logger.exception("--->")
    return output_dict


def get_actions(objectives: List[str],
                all_descriptor_df,
                limits,
                acquired_y: Dict[int, List[float]],
                living_compositions: List[str],
                random_seed=0,
                all_descriptor=True,
                n_descriptor_bayes=None,
                uses_permutation_importance=False,
                n_estimators=1000,
                num_probes=10,
                max_num_probes=1,
                n_rand_basis=0,
                blox=False):
    # all_x -> living_x, searched_x, searched_y
    # needs conversion map between index and compositions
    all_descriptor_df = copy.deepcopy(all_descriptor_df)
    living_descriptor_df = all_descriptor_df[all_descriptor_df["composition"].isin(living_compositions)]
    composition_col = all_descriptor_df.columns.get_loc("composition")
    living_x_with_compositions = living_descriptor_df.values
    logger.info("get_actions")
    logger.info("living_compositions")
    logger.info(living_compositions)
    comp_index_map = {x[composition_col]: int(n) for n, x in enumerate(living_x_with_compositions)}
    index_comp_map = {v: k for k, v in comp_index_map.items()}
    living_x = np.delete(living_x_with_compositions, obj=composition_col, axis=1)
    searched_compositions = sorted(list(acquired_y.keys()))
    searched_y = np.array([acquired_y[s] for s in searched_compositions])
    searched_index = list(comp_index_map[i] for i in searched_compositions)
    searched_x = living_x[searched_index]
    search_result = dict()
    n_objective = len(objectives)
    # Select descriptors
    if not all_descriptor:
        descriptor_indices, rfr_details = select_descriptors(n_descriptor_bayes, n_estimators, objectives,
                                                             searched_x, searched_y,
                                                             uses_permutation_importance)
        search_result["rfr_details"] = rfr_details
        search_result["descriptor_indices"] = descriptor_indices
    else:
        n_descriptor_bayes = living_x.shape[1]
        descriptor_indices = [list(range(n_descriptor_bayes)) for _ in range(n_objective)]
    living_x_reduced_descriptor = np.zeros((n_objective, len(living_x), n_descriptor_bayes))
    searched_x_reduced_descriptor = np.zeros((n_objective, len(searched_x), n_descriptor_bayes))
    for j in range(n_objective):
        living_x_reduced_descriptor[j] = copy.deepcopy(living_x[:, descriptor_indices[j]])  # 2021/9/7
        searched_x_reduced_descriptor[j] = copy.deepcopy(searched_x[:, descriptor_indices[j]])  # 2021/9/7

    # Standardize
    ss_x = [StandardScaler() for _ in range(n_objective)]
    for j in range(n_objective):
        ss_x[j].fit(searched_x_reduced_descriptor[j])
    ss_y = StandardScaler()
    ss_y.fit(searched_y[:, :])
    standardized_x_reduced_descriptor = \
        np.array([ss_x[j].transform(living_x_reduced_descriptor[j]) for j in range(n_objective)])
    standardized_y = np.array(ss_y.transform(searched_y))
    standardized_limit = np.array(ss_y.transform(np.array(limits).T)).T

    # Calculate priority
    if blox:
        if standardized_y.shape[1] != n_objective:
            msg = "ERROR: initial_data[1].shape[1] != num_objectives"
            raise RuntimeError(msg)
        if len(searched_index) != standardized_y.shape[0]:
            msg = "ERROR: len(initial_data[0]) != initial_data[1].shape[0]"
            raise RuntimeError(msg)
        test_index = sorted(list(set(np.arange(0, standardized_x_reduced_descriptor.shape[1])) - set(searched_index)))
        test_pred = np.zeros([n_objective, len(test_index)])
        for o in range(len(objectives)):
            n_cpus = multiprocessing.cpu_count()
            rfr = RandomForestRegressor(n_estimators=n_estimators,
                                        n_jobs=n_cpus,
                                        random_state=1)
            train_x = standardized_x_reduced_descriptor[o][searched_index]
            train_y = standardized_y[:, o]
            print("----------------------------")
            print(f"{train_y.shape=}")
            print(train_y)
            rfr.fit(train_x, train_y)
            test_x = standardized_x_reduced_descriptor[o][test_index]
            test_pred[o][:] = rfr.predict(test_x)
        sn = np.array([stein_novelty(test_p, standardized_y, sigma=0.1) for test_p in test_pred.T])
        actions = sn.argsort()[::-1][:num_probes]
        logger.info(f"Stein_novelty: {sn}")
    else:
        my_policy = Policy(standardized_x_reduced_descriptor,
                           num_objectives=n_objective,
                           initial_data=(searched_index, standardized_y),
                           comm=None)
        my_policy.set_seed(random_seed)
        actions = my_policy.bayes_search(max_num_probes=max_num_probes,
                                         num_search_each_probe=num_probes,
                                         simulator=None,
                                         score="RANGE",
                                         limit=standardized_limit,
                                         is_disp=False,
                                         interval=0,
                                         num_rand_basis=n_rand_basis)
    actions = [index_comp_map[i] for i in actions]  # convert to composition
    return actions


def set_priority_and_get_running_fw_ids(actions_by_comp: List[str],
                                        db,
                                        comp_fwid_map: Dict[str, List[int]]):
    lp = LaunchPad.auto_load()
    living_and_not_completed_fwids = lp.get_fw_ids_in_wfs(fw_query={"state": {"$in": ["RUNNING", "RESERVED"]}})
    running_fwids_priority = [(doc["fw_id"], doc["spec"]["_priority"]) for doc in
                              db["fireworks"].find({"fw_id": {"$in": living_and_not_completed_fwids}},
                                                   {"fw_id": True, "spec._priority": True})]
    running_fwids_priority.sort(key=lambda x: x[1], reverse=True)
    running_fw_ids = [i for i, _ in running_fwids_priority]
    #  Compositions to set priority
    #  make running wf continue with high priority
    compositions_to_set_priority = fwids_to_comps(running_fw_ids, comp_fwid_map)
    for action in actions_by_comp:
        if action not in compositions_to_set_priority:
            compositions_to_set_priority.append(action)
    priority_dict = {composition: priority
                     for priority, composition in enumerate(reversed(compositions_to_set_priority))}
    logger.info(f"{priority_dict=}")
    # Reset priority
    fw_ids = lp.get_fw_ids({"spec._priority": {"$ne": DEFAULT_PRIORITY}})
    for fw_id in fw_ids:
        lp.set_priority(fw_id, DEFAULT_PRIORITY)
    # Set priority
    for composition, priority in priority_dict.items():
        fw_ids = comp_fwid_map[composition]
        for fw_id in fw_ids:
            lp.set_priority(fw_id, priority)
    return list(set(running_fw_ids))


def select_descriptors(n_descriptor_bayes, n_estimators, objectives,
                       searched_x, searched_y, uses_permutation_importances):
    descriptor_indices = list()
    rfr_details = dict()
    rfr_details["importances"] = dict()
    if uses_permutation_importances:
        rfr_details["importances_each"] = list()
        rfr_details["importances_std"] = list()
    rfr_details["rfr_start"] = dict()
    rfr_details["rfr_end"] = dict()
    for o, obj in enumerate(objectives):
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
        logger.debug(f"searched_y:")
        logger.debug(f"{searched_y}")
        logger.debug(f"searched_y.shape:")
        logger.debug(f"{searched_y.shape}")
        searched_y_this_obj = searched_y[:, o]
        logger.debug(f"searched_y_this_obj")
        logger.debug(f"{searched_y_this_obj}")
        rfr.fit(searched_x, searched_y_this_obj)
        rfr_end = datetime.now()
        logger.info(f"random forest fit ends : {rfr_end} ({rfr_end - rfr_start} [sec])")
        if uses_permutation_importances:
            permutation_importance_result = permutation_importance(rfr, searched_x, searched_y_this_obj, n_jobs=-1)
            importances = permutation_importance_result["importances_mean"]
            rfr_details["importances_each"].append(permutation_importance_result["importances"])
            rfr_details["importances_std"].append(permutation_importance_result["importances_std"])
        else:
            importances = rfr.feature_importances_
        descriptor_indices.append(np.argsort(importances)[::-1][:n_descriptor_bayes])
        logger.debug("=" * 100)
        logger.debug(f"searched_x: {searched_x}")
        logger.debug("=" * 100)
        logger.debug(f"descriptor_indices: {descriptor_indices}")

        rfr_details["importances"][obj] = importances
        rfr_details["rfr_start"][obj] = rfr_start
        rfr_details["rfr_end"][obj] = rfr_end
    return descriptor_indices, rfr_details


def main():
    # Parse args
    args = get_args()
    mode_blox = args.blox
    objectives, limits = parse_objective(args.objective, mode_blox)
    conversion = args.conversion
    spec = importlib.util.spec_from_file_location("run_builder", args.builder)
    foo = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = foo
    spec.loader.exec_module(foo)
    run_builder = foo.run_builder
    limits = [[convert(lim, c) if lim is not None else None for lim in lim_obj]
              for lim_obj, c in zip(limits, conversion)]
    n_rand_basis = args.n_rand_basis
    max_num_probes = 1
    n_write_server = args.n_write_server
    random_seed = args.random_seed
    n_descriptor_bayes = args.n_descriptor
    all_descriptor = args.all_descriptor
    monitoring_period = args.monitor
    if args.all_descriptor:
        logger.warning("Ignore -n_descriptor")
    uses_permutation_importance = args.permutation_importance
    n_estimators = args.n_estimators
    descriptor_csv = args.descriptor_csv
    db_file = args.db_file
    restart_dir = args.restart_dir
    initial_priority = args.initial_priority
    if (restart_dir is not None) and (initial_priority is not None):
        raise ValueError(f"Both restart_dir and initial_priority can't be specified simultaneously.")

    # Prepare initial variables
    db = get_database(db_file)
    comp_fwids_map = get_comp_fwids_map(db)
    descriptor = pd.read_csv(descriptor_csv)
    # Second, descriptors are deleted if at least one material do not have those values.
    descriptors_to_delete = descriptor.isnull().any(axis=0)
    descriptor.drop(columns=descriptors_to_delete[descriptors_to_delete == True].index, inplace=True)
    dropped_descriptors: pd.Index = descriptors_to_delete[descriptors_to_delete == True].index
    logger.info(f"These {len(dropped_descriptors)} descriptors are dropped, "
                f"since at least one material doesn't have descriptor values. "
                f"(remaining: {len(descriptor.columns)})")
    logger.info(f"\n{dropped_descriptors}")

    unknown_compositions = descriptor["composition"].values.tolist()
    if n_write_server == -1:
        n_write_server = len(unknown_compositions)
    living_compositions = copy.deepcopy(unknown_compositions)
    acquired_properties = dict()

    for step in count():
        logger.info(f"---------step {step} begin-------------")
        datetime_begin = datetime.utcnow()
        logger.info(f"step: {step}")
        if restart_dir is None:
            os.mkdir(f"step_{step}")
        lp = LaunchPad.auto_load()
        new_acquired_properties = dict()
        discard_compositions = []
        if step == 0:
            if restart_dir is None:
                if initial_priority is None:
                    logger.info("Random search")
                    actions = random.sample(living_compositions, n_write_server)
                else:
                    with open(initial_priority, "r") as fr:
                        actions = json.load(fr)["actions"]
            else:
                restart_dir = Path(restart_dir)
                if "step_" not in restart_dir.stem:
                    restart_dir = max([Path(p) for p in restart_dir.glob("step_*")
                                       if (Path(p / "result.json")).exists()],
                                      key=lambda p: int(str(Path(p).stem).replace("step_", "")))
                logger.info(f"Restart from {restart_dir}")
                with open(restart_dir / "result.json") as fr:
                    result_dat = json.load(fr)
                actions = []  # fireworks would already have spec._priority
                acquired_properties = {int(k): v for k, v in result_dat["acquired_properties"].items()}
                discard_compositions = result_dat["discarded_compositions"]
        else:
            # When step = 1, at least two data is necessary for regression.
            known_compositions = list(acquired_properties.keys())
            while (step == 1 and len(acquired_properties) <= 1) or \
                    (step >= 2 and len(new_acquired_properties) == 0):
                run_builder(db_file)
                new_acquired_properties = get_materials_properties(db, objectives, known_compositions, conversion)
                acquired_properties.update(new_acquired_properties)
                logger.info(f"{acquired_properties=}")
                logger.info(f"Waiting for new acquired properties, sleeping {monitoring_period} sec... (step: {step})")
                time.sleep(monitoring_period)
                lp.detect_lostruns(rerun=True)
                lp.detect_unreserved(rerun=True)
            discard_compositions = get_fizzled_composition(known_compositions, db, comp_fwids_map)
            logger.info("discarded compositions")
            logger.info(discard_compositions)
            for i in discard_compositions:
                if i in living_compositions:
                    logger.info(f"discarded {i}")
                    living_compositions.remove(i)
            actions = get_actions(objectives,
                                  descriptor,
                                  limits,
                                  acquired_properties,
                                  living_compositions,
                                  random_seed=random_seed,
                                  all_descriptor=all_descriptor,
                                  n_descriptor_bayes=n_descriptor_bayes,
                                  uses_permutation_importance=uses_permutation_importance,
                                  n_estimators=n_estimators,
                                  num_probes=n_write_server,
                                  max_num_probes=max_num_probes,
                                  n_rand_basis=n_rand_basis,
                                  blox=mode_blox
                                  )
        running_fw_ids = set_priority_and_get_running_fw_ids(actions, db, comp_fwids_map)
        datetime_end = datetime.utcnow()
        result = {
            "datetime_begin": datetime_begin.isoformat(),
            "datetime_end": datetime_end.isoformat(),
            "actions": actions,
            "acquired_properties": acquired_properties,
            "new_acquired_properties": new_acquired_properties,
            "discarded_compositions": discard_compositions,
            "living_compositions": living_compositions,
            "running_fw_ids": running_fw_ids,
            "mode_blox": mode_blox
        }
        with open(f"step_{step}/result.json", "w") as fw:
            json.dump(result, fw, indent=4)


if __name__ == "__main__":
    main()
