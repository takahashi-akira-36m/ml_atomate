from typing import List, Optional, Iterable, Sequence
import numpy as np
import pandas as pd
import importlib
from enum import Enum
import os
import sys
import json
from datetime import datetime
import traceback
from collections import defaultdict
from itertools import chain
from fireworks import LaunchPad
from logging import getLogger
from atomate.utils.utils import get_database
from ml_atomate.utils.util import get_from_mongo_like_str
from ml_atomate import __version__

logger = getLogger(__name__)


class State(Enum):
    completed = "COMPLETED"
    waiting = "WAITING"
    fizzled = "FIZZLED"

    def __str__(self):
        return self.value


# index composition -> formula_pretty
# TODO: use arbitrary structure (and thereby enable non-unique composition) and index
class DataManager:
    DEFAULT_PRIORITY = -1

    def __init__(self, compositions: Sequence[str], objective_paths: Sequence[str],
                 builder_file: str, db_file: str, index_name: Optional[str] = None):
        # TODO
        if index_name != "composition":
            raise ValueError("Now index can be specified by only composition")
        if "state" in objective_paths:
            raise ValueError("state cannot be used as column name")
        column_names = list(objective_paths) + ["state"]
        if index_name is not None:
            column_names = [index_name] + column_names
        self._data_frame = pd.DataFrame(columns=column_names)
        self._data_frame[index_name] = compositions
        self._data_frame.set_index(index_name, inplace=True)
        for col in objective_paths:
            self._data_frame[col] = None
        self._data_frame["state"] = str(State.waiting)
        self._objective_paths = list(objective_paths)
        self._db_file = db_file
        db = get_database(db_file)
        self._db = db
        # Builder
        spec = importlib.util.spec_from_file_location("run_builder", builder_file)
        foo = importlib.util.module_from_spec(spec)
        sys.modules["module.name"] = foo
        spec.loader.exec_module(foo)
        self._run_builder_func = foo.run_builder
        # Setting
        logger.info("Get map of fw_id and composition...")
        fw_col = db["fireworks"]
        dd = defaultdict(list)
        for doc in fw_col.find({}, ["name", "fw_id"]):
            # name: {composition}-{task_label}
            comp = doc["name"].split("-")[0]
            dd[comp].append(doc["fw_id"])
        self._comp_fwids_map = dd

    def dump_search_step_data(self, dir_name: str):
        self._data_frame.to_csv(f"{dir_name}/obj_data_frame.csv")
        d = {
            "objective_paths": self._objective_paths,
        }
        with open(f"{dir_name}/df_col.json", "w") as fw:
            json.dump(d, fw)
        d = {
            "version": __version__,
            "datetime": str(datetime.utcnow())
        }
        with open(f"{dir_name}/meta_data.json", "w") as fw:
            json.dump(d, fw)

    @classmethod
    def read_search_step_data(cls, dir_name: str, db_file: str, builder_file: str,
                              descriptor_csv: Optional[str] = None, objective_paths: Optional[List[str]] = None,
                              index_name: str = None):
        # Old version
        if not os.path.exists(f"{dir_name}/obj_data_frame.csv") and os.path.exists(f"{dir_name}/result.json"):
            obj = cls(pd.read_csv(descriptor_csv).index, objective_paths, db_file)
            with open(f"{dir_name}/result.json") as fr:
                acquired_properties = json.load(fr)["acquired_properties"]
                fizzled_compositions = json.load(fr)["discarded_compositions"]
            for _, comp in obj._data_frame.iterrows():
                if comp in acquired_properties:
                    for i, p in enumerate(objective_paths):
                        obj._data_frame[comp][p] = acquired_properties[comp][i]
                if comp in fizzled_compositions:
                    obj._data_frame[comp]["state"] = str(State.fizzled)
            return obj
        else:
            df = pd.read_csv(f"{dir_name}/obj_data_frame.csv")
            with open(f"{dir_name}/df_col.json") as fr:
                d = json.load(fr)
            if index_name is not None:
                df.set_index(index_name, inplace=True)
            return cls(
                compositions=df.index.tolist(),
                objective_paths=d["objective_paths"],
                builder_file=builder_file,
                db_file=db_file,
                index_name=index_name
            )

    @property
    def index_name(self):
        return self._data_frame.index.name

    @property
    def all_index(self) -> pd.Index:
        return self._data_frame.index

    @property
    def fizzled_index(self) -> pd.Index:
        return self.fizzled_df.index

    @property
    def not_fizzled_index(self) -> pd.Index:
        return self.not_fizzled_df.index

    @property
    def not_fizzled_df(self) -> pd.DataFrame:
        return self._data_frame[self._data_frame["state"] != str(State.fizzled)]

    @property
    def fizzled_df(self) -> pd.DataFrame:
        return self._data_frame[self._data_frame["state"] == str(State.fizzled)]

    @property
    def objective_paths(self) -> List[str]:
        return self._objective_paths

    @property
    def searched_df(self) -> pd.DataFrame:
        return self._data_frame[self._data_frame["state"] == str(State.completed)]

    @property
    def candidate_df(self) -> pd.DataFrame:
        return self._data_frame.loc[~self._data_frame["state"].isin([str(State.completed), str(State.fizzled)])]

    @property
    def searched_index(self) -> pd.Index:
        return self.searched_df.index

    @property
    def candidate_index(self) -> pd.Index:
        return self.candidate_df.index

    @property
    def all_data_frame(self) -> pd.DataFrame:
        return self._data_frame

    @property
    def comp_fwids_map(self):
        return self._comp_fwids_map

    @property
    def db(self):
        return self._db

    def comps_to_fwids(self, comp_array: Iterable[str]):
        return list(chain.from_iterable(self._comp_fwids_map[comp] for comp in comp_array))

    def fwids_to_comps(self, fw_id_array: List[int]):
        return [comp for comp, fw_ids in self._comp_fwids_map.items() if set(fw_id_array) & set(fw_ids)]

    def dump_csv_file(self, file_name: str):
        self._data_frame.to_csv(file_name)

    def update_from_database(self):  # if returns true, at least one data is updated
        fizzled_updated = self._get_fizzled_composition()
        self._run_builder_func(self._db_file)
        return fizzled_updated, self._get_materials_properties()

    # if returns true, at least one data is updated
    def _get_materials_properties(self) -> bool:  # TODO: implement debug mode and test
        is_updated = False
        query = {p: {"$exists": True} for p in self.objective_paths}
        query["formula_pretty"] = {"$nin": list(self.searched_index)}
        target_compositions = \
            list(doc["formula_pretty"] for doc in self._db["materials"].find(query, {"formula_pretty": True}))
        for composition in target_compositions:
            projection = {".".join([k for k in p.split(".") if not k.isdecimal()]): True for p in self.objective_paths}
            if self.index_name == "composition":
                query = {"formula_pretty": composition}
            else:
                raise ValueError("Currently index can be specified by only composition")
            doc = self._db["materials"].find_one(query, projection)
            if doc is not None:
                try:
                    for p in self.objective_paths:
                        self._data_frame.loc[composition, p] = get_from_mongo_like_str(doc, p)
                except:
                    logger.exception("<---")
                    logger.exception(f"There was an error processing composition: {composition}")
                    logger.exception(traceback.format_exc())
                    logger.exception("--->")
            if all(not np.isnan(self._data_frame.loc[composition, p]) for p in self.objective_paths):
                is_updated = True
                self._data_frame.loc[composition, "state"] = str(State.completed)
        return is_updated

    # if returns true, at least one data is updated
    def _get_fizzled_composition(self) -> bool:
        is_updated = False
        already_known_fw_ids = self.comps_to_fwids(self.fizzled_index)
        fizzled_fw_ids = \
            set(fizzled_doc["fw_id"]
                for fizzled_doc in self._db["fireworks"].find({"state": "FIZZLED",
                                                               "fw_id": {"$nin": already_known_fw_ids}}, ["fw_id"]))
        for comp, fw_ids in self._comp_fwids_map.items():
            if set(fizzled_fw_ids) & set(fw_ids):
                is_updated = True
                self._data_frame.loc[comp, "state"] = str(State.fizzled)
        return is_updated

    def get_running_comps(self) -> list[str]:
        lp = LaunchPad.auto_load()
        running_fwids = lp.get_fw_ids_in_wfs(fw_query={"state": {"$in": ["RUNNING", "RESERVED"]}})
        running_fwids_priority = [(doc["fw_id"], doc["spec"]["_priority"]) for doc in
                                  self.db["fireworks"].find({"fw_id": {"$in": running_fwids}},
                                                            {"fw_id": True, "spec._priority": True})]
        running_fwids_priority = sorted(running_fwids_priority, key=lambda x: x[1], reverse=True)
        return [k for i, _ in running_fwids_priority for k, v in self.comp_fwids_map.items()
                if (k in self.candidate_index and i in v)]

    def set_priority(self,
                     actions: List[str],  # if ["MgO", "BaO", "CaO"], then priority = MgO:3, BaO:2, CaO:1, the others:-1
                     stop_running_fw: bool = False):
        lp = LaunchPad.auto_load()
        if stop_running_fw:
            compositions_to_set_priority = []
        else:
            compositions_to_set_priority = self.get_running_comps()
        #  Compositions to set priority
        #  make running wf continue with high priority
        for action in actions:
            if action not in compositions_to_set_priority:
                compositions_to_set_priority.append(action)
        priority_dict = {composition: priority
                         for priority, composition in enumerate(reversed(compositions_to_set_priority))}
        logger.info(f"{priority_dict=}")
        # Reset priority
        fw_ids = lp.get_fw_ids({"spec._priority": {"$ne": self.DEFAULT_PRIORITY}})
        for fw_id in fw_ids:
            lp.set_priority(fw_id, self.DEFAULT_PRIORITY)
        # Set priority
        for composition, priority in priority_dict.items():
            fw_ids = self.comp_fwids_map[composition]
            for fw_id in fw_ids:
                lp.set_priority(fw_id, priority)

