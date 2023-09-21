import argparse
from collections import defaultdict
from datetime import datetime
from typing import Dict, List
from atomate.utils.utils import get_database
import random
import json
from logging import getLogger, DEBUG, basicConfig
from pymongo import database
from fireworks import LaunchPad


logger = getLogger(__name__)
now = datetime.now()
FW_LOGGING_FORMAT = '%(asctime)s %(levelname)s %(message)s'  # format for loggers
basicConfig(filename=f"priority_{now.year}{now.month}{now.day}_{now.hour}{now.minute}{now.second}{now.microsecond}.log",
            format=FW_LOGGING_FORMAT)
logger.setLevel(DEBUG)


def get_comp_fwids_map(db: database) -> Dict[str, List[int]]:
    logger.info("Get map of fw_id and composition...")
    fw_col = db["fireworks"]
    dd = defaultdict(list)
    for doc in fw_col.find({}, ["name", "fw_id"]):
        # name: {compostiion}-{task_label}
        comp = doc["name"].split("-")[0]
        dd[comp].append(doc["fw_id"])
    return dd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_file", "-df",
                        help="path to db.json",
                        type=str,
                        required=True)
    parser.add_argument("--initial_priority", "-ip",
                        help="Specify step_*/result.json. Same initial priority will be set",
                        default=None,
                        type=str)
    parser.add_argument("--seed", "-s",
                        help="Random seed",
                        default=None,
                        type=str)
    parser.add_argument("--src_db_file", "-sdf",
                        help="path to db.json to copy priorities",
                        default=None,
                        type=str)
    args = parser.parse_args()
    return args


def main():
    lp = LaunchPad.auto_load()
    args = get_args()
    db_file = args.db_file
    db = get_database(db_file)
    comp_fwids_map = get_comp_fwids_map(db)
    ip = args.initial_priority
    src_db_file = args.src_db_file
    seed = args.seed
    if src_db_file is None:
        if ip is None:
            actions = []
        else:
            with open(ip, "r") as fr:
                actions = json.load(fr)["actions"]
        if seed is not None:
            random.seed(seed)
        rest_comps = list(set(comp_fwids_map.keys()) - set(actions))
        random.shuffle(rest_comps)
        actions = actions + rest_comps
        print(actions)
        priority_dict = {composition: priority
                         for priority, composition in enumerate(reversed(actions))}
        for composition, priority in priority_dict.items():
            fw_ids = comp_fwids_map[composition]
            for fw_id in fw_ids:
                lp.set_priority(fw_id, priority)
    else:
        db_src = get_database(src_db_file)
        priority_dict = {doc["name"].split("-")[0]: doc["spec"]["_priority"]
                         for doc in db_src["fireworks"].find({}, {"spec": True, "name": True})}
        inv_priority_dict = {v: k for k, v in priority_dict.items()}
        actions = []
        for i in range(max(priority_dict.values()), min(priority_dict.values()) - 1, -1):
            actions.append(inv_priority_dict[i])
        with open("initial_priority.json", "w") as fw:
            json.dump({"actions": actions}, fw)


if __name__ == "__main__":
    main()
