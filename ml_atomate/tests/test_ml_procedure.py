import json
import os
import unittest
import sys
import pandas as pd
import numpy as np
from ruamel.yaml import YAML
from typing import List
from fireworks import LaunchPad
from fireworks.utilities.fw_serializers import reconstitute_dates
from traceback import print_exc
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin

from physbo.search.discrete.policy import policy as policy_discrete
from physbo.search.discrete_multi.policy import policy as policy_discrete_multi
from ml_atomate.data_manager import DataManager
from ml_atomate.ml_procedure import MlProcedure, Conversion
from ml_atomate.physbo_customized.policy_ptr import policy_ptr
from ml_atomate.utils.util import TEST_DB_DIR


class DummyModel(BaseEstimator, RegressorMixin):
    def __init__(self, i):
        self.i = i

    def fit(self, x, y):
        pass

    def predict(self, x):
        return x[:, self.i]


def test_x(x: List[float], scale: float) -> float:
    val = sum(x) * scale
    return val


class TestMlProcedure(unittest.TestCase):
    TEST_DIR = f"{os.path.dirname(__file__)}/data/hse_diel_test/"
    DB_DIR = TEST_DB_DIR
    if not os.path.exists(DB_DIR):
        raise ValueError("Set TEST_DB_DIR")

    @classmethod
    def setUpClass(cls) -> None:
        try:
            # cls.lp: LaunchPad = LaunchPad.from_file(filename=f"{cls.DB_DIR}/my_launchpad.yaml")
            # Since error raised when used LaunchPad.from_file, I copied and rewrited safe_load
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
            cls.raw_descriptor_df = descriptor_df
            objective_paths = ("bandstructure_hse.bandgap", "dielectric.epsilon_avg")
            builder_file = f"{cls.TEST_DIR}/run_builder.py"
            db_file = f"{cls.DB_DIR}/db.json"
            cls.data_manager = DataManager(descriptor_df["composition"].tolist(), objective_paths, builder_file,
                                           db_file, index_name="composition")
            cls.data_manager.update_from_database()
            cls.ml_proc = MlProcedure.from_data_manager(cls.data_manager, descriptor_dataframe=descriptor_df,
                                                        conversion=["no_conversion", Conversion.log])
            # Single for physbo.policy_discrete
            cls.data_manager_single = DataManager(descriptor_df["composition"].tolist(),
                                                  ["bandstructure_hse.bandgap"], builder_file, db_file,
                                                  index_name="composition")
            cls.data_manager_single.update_from_database()
            cls.ml_proc_single = MlProcedure.from_data_manager(cls.data_manager_single,
                                                               descriptor_dataframe=descriptor_df)

            # For prune descriptor
            cls.data_manager_prune = DataManager(descriptor_df["composition"].tolist(), objective_paths, builder_file,
                                                 db_file, index_name="composition")
            cls.data_manager_prune.update_from_database()
            cls.ml_proc_prune = MlProcedure.from_data_manager(cls.data_manager_prune, descriptor_dataframe=descriptor_df,
                                                              conversion=[Conversion.no_conversion, Conversion.log])

        except Exception:
            print_exc()
            raise unittest.SkipTest("Failed to setup MongoDB. Skipping tests.")

    @classmethod
    def tearDownClass(cls):
        pass
        # cls.lp.reset(password=None, require_password=False)

    def test_properties(self):
        self.assertSetEqual(set(self.ml_proc.searched_index), {"MgO", "CaO"})
        self.assertSetEqual(set(self.ml_proc.candidate_index), {"BaO", "SrO"})
        self.assertSetEqual(set(self.ml_proc.descriptor_col), {"x1", "x3"})
        self.assertListEqual(self.ml_proc.objective_col, ["bandstructure_hse.bandgap", "dielectric.epsilon_avg"])
        # train_x, Value is standardized
        self.assertSetEqual(set(self.ml_proc.train_x.keys()), set(self.ml_proc.objective_col))
        self.assertSetEqual(set(self.ml_proc.train_x["bandstructure_hse.bandgap"].index),
                            set(self.ml_proc.searched_index))
        self.assertAlmostEqual(self.ml_proc.train_x["bandstructure_hse.bandgap"].loc["MgO", "x1"], -1.230914909793327)
        self.assertAlmostEqual(self.ml_proc.train_x["dielectric.epsilon_avg"].loc["MgO", "x1"], -1.230914909793327)
        # candidate_x, Value is standardized
        self.assertSetEqual(set(self.ml_proc.candidate_x.keys()), set(self.ml_proc.objective_col))
        self.assertSetEqual(set(self.ml_proc.candidate_x["bandstructure_hse.bandgap"].index),
                            set(self.ml_proc.candidate_index))
        self.assertAlmostEqual(self.ml_proc.candidate_x["bandstructure_hse.bandgap"].loc["BaO", "x1"], -0.492366)
        self.assertAlmostEqual(self.ml_proc.candidate_x["dielectric.epsilon_avg"].loc["BaO", "x1"], -0.492366)
        # train_y, Value is standardized
        self.assertSetEqual(set(self.ml_proc.train_y.columns), set(self.ml_proc.objective_col))
        self.assertSetEqual(set(self.ml_proc.train_y["bandstructure_hse.bandgap"].index),
                            set(self.ml_proc.searched_index))
        self.assertAlmostEqual(self.ml_proc.train_y.loc["MgO", "bandstructure_hse.bandgap"], 1.0)
        self.assertAlmostEqual(self.ml_proc.train_y.loc["CaO", "dielectric.epsilon_avg"], 1.0)

        # TODO: create test
        self.assertEqual(self.ml_proc.selected_descriptors, None)

        self.assertAlmostEqual(self.ml_proc.scaler['x1'].mean_[0], 2)
        self.assertAlmostEqual(self.ml_proc.scaler['x1'].var_[0], 16.5)
        self.assertAlmostEqual(self.ml_proc.scaler['x3'].mean_[0], 4)
        self.assertAlmostEqual(self.ml_proc.scaler['x3'].var_[0], 16.5)
        self.assertAlmostEqual(self.ml_proc.scaler['bandstructure_hse.bandgap'].mean_[0], 5.87705)
        self.assertAlmostEqual(self.ml_proc.scaler['bandstructure_hse.bandgap'].var_[0], 0.313656)
        self.assertAlmostEqual(self.ml_proc.scaler['dielectric.epsilon_avg'].mean_[0], 1.07322996)
        self.assertAlmostEqual(self.ml_proc.scaler['dielectric.epsilon_avg'].var_[0], 0.00595191)
        self.assertEqual(str(self.ml_proc.conversion['bandstructure_hse.bandgap']), "Conversion.no_conversion")
        self.assertEqual(str(self.ml_proc.conversion['dielectric.epsilon_avg']), "Conversion.log")

    def test_transform(self):
        mgo_raw_gap = self.data_manager.all_data_frame.loc["MgO", "bandstructure_hse.bandgap"]
        expected = self.ml_proc.train_y.loc["MgO", "bandstructure_hse.bandgap"]
        actual = self.ml_proc.transform_param(mgo_raw_gap, "bandstructure_hse.bandgap")
        self.assertAlmostEqual(expected, actual)
        # Test conversion log
        mgo_raw_gap = self.data_manager.all_data_frame.loc["MgO", "dielectric.epsilon_avg"]
        expected = self.ml_proc.train_y.loc["MgO", "dielectric.epsilon_avg"]
        actual = self.ml_proc.transform_param(mgo_raw_gap, "dielectric.epsilon_avg")
        self.assertAlmostEqual(expected, actual)
        mgo_raw_x1 = self.raw_descriptor_df.set_index("composition").loc["MgO", "x1"]
        expected = self.ml_proc.train_x["bandstructure_hse.bandgap"].loc["MgO", "x1"]
        actual = self.ml_proc.transform_param(mgo_raw_x1, "x1")
        self.assertAlmostEqual(expected, actual)
        expected = np.NaN
        actual = self.ml_proc.transform_param(np.NaN, "dielectric.epsilon_avg")
        self.assertTrue(expected is actual)

    def test_prune_descriptors(self):
        # Only check number and columns are descriptors
        self.ml_proc_prune.prune_descriptors_rf_importance(
            n_estimators=100,
            n_descriptor_bayes=1,
            uses_permutation_importances=False)
        self.assertTrue(all(len(df.columns) == 1 for df in self.ml_proc_prune.train_x.values()))
        self.assertTrue(all(set(df.columns) < set(self.ml_proc.train_x["bandstructure_hse.bandgap"].columns)
                            for df in self.ml_proc_prune.train_x.values()))

    def test_get_action(self):
        # sklearn
        actions = self.ml_proc.get_priority(model_type=DummyModel, acquisition_function=test_x,
                                            model_params={"i": 0},
                                            acquisition_params={"scale": 1}, num_probes=2
                                            )
        self.assertListEqual(actions, ["SrO", "BaO"])
        actions = self.ml_proc.get_priority(model_type=DummyModel, acquisition_function=test_x,
                                            model_params={"i": 0},
                                            acquisition_params={"scale": -1}, num_probes=2
                                            )
        self.assertListEqual(actions, ["BaO", "SrO"])

        # TODO: This tests ensures only whether actions are not-searched compositions but not order
        # sklearn RF and BLOX
        self.ml_proc.get_priority(model_type=RandomForestRegressor, acquisition_function=self.ml_proc.stein_novelty,
                                  model_params={"n_estimators": 100},
                                  acquisition_params={"sigma": 0.1}, num_probes=2
                                  )

        # original physbo policy
        actions = self.ml_proc_single.get_priority(model_type=policy_discrete, acquisition_function=None,
                                                   model_params={"simulator": None, "seed": 42},
                                                   acquisition_params={"score": "PI"}, num_probes=2)
        self.assertSetEqual(set(self.ml_proc_single.candidate_index), set(actions))

        # When num_probes != 1, show_start_message_multi_search_mo raises error
        actions = self.ml_proc.get_priority(model_type=policy_discrete_multi, acquisition_function=None,
                                            model_params={"simulator": None, "seed": 1},
                                            acquisition_params={"score": "HVPI"}, num_probes=1)
        self.assertTrue(set(self.ml_proc.candidate_index) > set(actions))

        # ptr_physbo
        raw_limit = [[4, float("nan")], [30, float("nan")]]
        limit = [tuple(self.ml_proc.transform_param(l_val, o) for l_val in lim)
                 for o, lim in zip(self.ml_proc.objective_col, raw_limit)]
        actions = self.ml_proc.get_priority(model_type=policy_ptr, acquisition_function=None,
                                            model_params={"simulator": None, "seed": None},
                                            acquisition_params={"score": "RANGE",
                                                               "limit": limit}, num_probes=2)
        self.assertSetEqual(set(self.ml_proc.candidate_index), set(actions))


if __name__ == "__main__":
    unittest.main()
