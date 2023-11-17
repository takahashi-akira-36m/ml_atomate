import argparse
from datetime import datetime
from pathlib import Path
import random
import json
import os
import re
import time
from logging import getLogger, DEBUG, basicConfig
import pandas as pd
from fireworks import LaunchPad

from sklearn.ensemble import RandomForestRegressor
from ml_atomate.data_manager import DataManager
from ml_atomate.ml_procedure import MlProcedure
from ml_atomate.utils.util import parse_objective
from ml_atomate.physbo_customized.policy_ptr import policy_ptr


logger = getLogger(__name__)
now = datetime.now()
FW_LOGGING_FORMAT = '%(asctime)s %(levelname)s %(message)s'  # format for loggers
basicConfig(filename=f"priority_{now.year}{now.month}{now.day}_{now.hour}{now.minute}{now.second}{now.microsecond}.log",
            format=FW_LOGGING_FORMAT)
logger.setLevel(DEBUG)


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

    parser.add_argument("--sigma", "-s",
                        help="Sigma of blox",
                        default=0.1)

    parser.add_argument("--initial_priority", "-ip",
                        help="Specify step_*/result.json. Same initial priority will be set",
                        type=str)

    args = parser.parse_args()
    return args


def main():
    # Parse args
    args = get_args()
    mode_blox = args.blox
    objectives, limits = parse_objective(args.objective, mode_blox)
    conversion = args.conversion
    builder_file = args.builder
    n_rand_basis = args.n_rand_basis
    sigma = args.sigma
    max_num_probes = 1
    n_write_server = args.n_write_server
    physbo_seed = args.random_seed
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

    step = 0
    while True:
        logger.info(f"---------step {step} begin-------------")
        logger.info(f"step: {step}")
        if restart_dir is None:
            os.mkdir(f"step_{step}")
        lp = LaunchPad.auto_load()
        if step == 0:
            if restart_dir is None:
                data_manager = DataManager(
                    compositions=pd.read_csv(descriptor_csv)["composition"],
                    objective_paths=objectives,
                    builder_file=builder_file,
                    db_file=db_file,
                    index_name="composition"
                )
                data_manager.update_from_database()
                if n_write_server == -1:
                    n_write_server = len(data_manager.all_index)

                if initial_priority is None:
                    logger.info("Random search")
                    priority = random.sample(list(data_manager.all_index), n_write_server)
                else:
                    with open(initial_priority, "r") as fr:
                        priority = json.load(fr)["priority"]
            else:
                restart_dir = Path(restart_dir)
                if "step_" not in restart_dir.stem:
                    restart_dir = max([Path(p) for p in restart_dir.glob("step_*")
                                       if (Path(p / "obj_data_frame.csv")).exists()],
                                      key=lambda p: int(str(Path(p).stem).replace("step_", "")))
                logger.info(f"Restart from {restart_dir}")
                step = int(re.findall(r'\d+', str(restart_dir))[-1])
                logger.info(f"Now step is {step}")
                # TODO enable arbitrary index_name
                data_manager = DataManager.read_search_step_data(str(restart_dir), db_file, builder_file,
                                                                 index_name="composition")
                data_manager.update_from_database()
                try:
                    with open(restart_dir / "priority.json") as fr:
                        priority = json.load(fr)["priority"]
                except FileNotFoundError:
                    with open(restart_dir / "result.json") as fr:
                        priority = json.load(fr)["priority"]
        else:
            # When step = 1, at least two data is necessary for regression.
            is_updated = False
            while (step == 1 and len(data_manager.searched_index) <= 1) or \
                    (step >= 2 and not is_updated):
                _, is_updated = data_manager.update_from_database()
                logger.info(f"Waiting for new acquired properties, sleeping {monitoring_period} sec... (step: {step})")
                time.sleep(monitoring_period)
                lp.detect_lostruns(rerun=True)
                lp.detect_unreserved(rerun=True)
            ml_proc = MlProcedure.from_data_manager(data_manager, pd.read_csv(descriptor_csv), conversion=conversion)
            # PTR
            transformed_limits = [[ml_proc.transform_param(lim, o) if lim is not None else None for lim in lim_obj]
                                  for lim_obj, o in zip(limits, ml_proc.objective_col)]
            if mode_blox:
                model_type = RandomForestRegressor
                model_params = {"n_estimators": n_estimators}
                acquisition_function = ml_proc.stein_novelty
                acquisition_params = {"sigma": sigma}
            else:
                model_type = policy_ptr
                model_params = {"simulator": None,
                                "num_rand_basis": n_rand_basis,
                                "max_num_probes": max_num_probes,
                                "seed": physbo_seed}
                acquisition_function = None
                acquisition_params = {"score": "RANGE",
                                      "limit": transformed_limits}
            if not all_descriptor:
                ml_proc.prune_descriptors_rf_importance(n_descriptor_bayes,
                                                        n_estimators,
                                                        uses_permutation_importance)
            priority = ml_proc.get_priority(
                model_type=model_type,
                model_params=model_params,
                acquisition_params=acquisition_params,
                acquisition_function=acquisition_function,
                num_probes=n_write_server)
        data_manager.set_priority(priority, stop_running_fw=False)
        data_manager.dump_search_step_data(f"step_{step}/")
        with open(f"step_{step}/priority.json", "w") as fw:
            json.dump({"priority": priority}, fw)
        step += 1


if __name__ == "__main__":
    main()
