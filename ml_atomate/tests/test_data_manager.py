import json
import os
import unittest
import pandas as pd
import sys
from fireworks import LaunchPad
from traceback import print_exc
from ruamel.yaml import YAML
from fireworks.utilities.fw_serializers import reconstitute_dates

from ml_atomate.data_manager import DataManager
from ml_atomate.utils.util import TEST_DB_DIR


class TestDataManager(unittest.TestCase):
    TEST_DIR = f"{os.path.dirname(__file__)}/data/hse_diel_test/"
    DB_DIR = TEST_DB_DIR
    if not os.path.exists(DB_DIR):
        raise ValueError("Set TEST_DB_DIR")

    @classmethod
    def setUpClass(cls) -> None:
        try:
            if sys.version_info > (3, 0, 0):
                ENCODING_PARAMS = {"encoding": "utf-8"}
            else:
                ENCODING_PARAMS = {}
            with open(f"{cls.DB_DIR}/my_launchpad.yaml", 'r', **ENCODING_PARAMS) as f:
                dct = YAML().load(f.read())  # edited from original fw
            cls.lp = LaunchPad.from_dict(reconstitute_dates(dct))
            cls.lp.reset(password=None, require_password=False)
            db = cls.lp.db
            fireworks_json = f"{cls.TEST_DIR}/fireworks.json"
            with open(fireworks_json) as fr:
                docs = json.load(fr)
            db["fireworks"].insert_many([{k: v for k, v in d.items() if k != "_id"} for d in docs])
            tasks_json = f"{cls.TEST_DIR}/tasks.json"
            with open(tasks_json) as fr:
                docs = json.load(fr)
            db["tasks"].insert_many([{k: v for k, v in d.items() if k != "_id"} for d in docs])
            descriptor_df = pd.read_csv(f"{cls.TEST_DIR}/descriptors.csv")
            objective_paths = ["bandstructure_hse.bandgap", "dielectric.epsilon_avg"]
            builder_file = f"{cls.TEST_DIR}/run_builder.py"
            db_file = f"{cls.DB_DIR}/db.json"
            cls.data_manager = DataManager(descriptor_df["composition"].tolist(), objective_paths, builder_file, db_file,
                                           index_name="composition")
        except Exception:
            print_exc()
            raise unittest.SkipTest("Failed to setup MongoDB. Skipping tests.")

    @classmethod
    def tearDownClass(cls):
        pass

    def test_properties(self):
        self.assertEqual("composition", self.data_manager.index_name)
        self.assertSetEqual(set(self.data_manager.all_index), {"MgO", "CaO", "SrO", "BaO", "TsO", "OgO"})
        self.assertListEqual(self.data_manager.objective_paths, ["bandstructure_hse.bandgap", "dielectric.epsilon_avg"])
        self.assertDictEqual(self.data_manager.comp_fwids_map, {"MgO": [1, 2, 3],
                                                                "BaO": [4, 5, 6],
                                                                "CaO": [7, 8, 9],
                                                                "OgO": [10, 11, 12],
                                                                "SrO": [13, 14, 15],
                                                                "TsO": [16, 17, 18]})
        fizzled_updated, property_updated = self.data_manager.update_from_database()
        self.assertEqual(fizzled_updated, True)
        self.assertEqual(property_updated, True)
        self.assertSetEqual(set(self.data_manager.fizzled_index), {"TsO", "OgO"})
        self.assertSetEqual(set(self.data_manager.not_fizzled_index), {"MgO", "CaO", "SrO", "BaO"})
        searched_df = self.data_manager.searched_df
        self.assertAlmostEqual(searched_df.loc["MgO", "bandstructure_hse.bandgap"], 6.4371)
        self.assertAlmostEqual(searched_df.loc["MgO", "dielectric.epsilon_avg"], 9.9101755)
        self.assertAlmostEqual(searched_df.loc["CaO", "bandstructure_hse.bandgap"], 5.3170)
        self.assertAlmostEqual(searched_df.loc["CaO", "dielectric.epsilon_avg"], 14.137694)
        not_searched_df = self.data_manager.candidate_df
        self.assertSetEqual(set(not_searched_df.index), {"SrO", "BaO"})
        fizzled_updated, property_updated = self.data_manager.update_from_database()
        self.assertEqual(fizzled_updated, False)
        self.assertEqual(property_updated, False)
        # TODO: set_priority, dump_load


if __name__ == "__main__":
    unittest.main()
