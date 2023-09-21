import json
import os
import unittest
import math
from pymongo import MongoClient
from monty.serialization import loadfn
from fireworks import LaunchPad


from ml_atomate.priority_setter import get_comp_fwids_map, comps_to_fwids, fwids_to_comps, val_in_limit, \
    get_fizzled_composition, convert, get_materials_properties


class TestPrioritySetter(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.lp: LaunchPad = LaunchPad.from_file(filename="/home/taka/fw_config/my_launchpad.yaml")
            cls.lp.reset(password=None, require_password=False)
            cls.db = cls.lp.db
            fireworks_json = f"{os.path.dirname(__file__)}/data/fireworks.json"
            with open(fireworks_json) as fr:
                docs = json.load(fr)
            cls.db["fireworks"].insert_many(docs)
            materials_json = f"{os.path.dirname(__file__)}/data/materials.json"
            with open(materials_json) as fr:
                docs = json.load(fr)
            cls.db["materials"].insert_many(docs)
        except Exception:
            raise unittest.SkipTest("Failed to setup MongoDB. Skipping tests.")

    @classmethod
    def tearDownClass(cls):
        cls.lp.reset(password=None, require_password=False)

    def test_get_comp_fwids_map(self):
        comp_fwids_map = get_comp_fwids_map(self.db)
        self.assertDictEqual(comp_fwids_map, {"SbO2": [10, 11, 12], "SrO2": [394, 395, 396]})
        self.assertSetEqual(set(comps_to_fwids(["SrO2"], comp_fwids_map)), {394, 395, 396})
        self.assertSetEqual(set(comps_to_fwids(["SrO2", "SbO2"], comp_fwids_map)), {10, 11, 12, 394, 395, 396})
        self.assertSetEqual(set(fwids_to_comps([10, 11, 12], comp_fwids_map)), {"SbO2"})
        self.assertSetEqual(set(fwids_to_comps([10, 11, 12, 394, 395, 396], comp_fwids_map)), {"SbO2", "SrO2"})

    def test_val_in_limit(self):
        self.assertTrue(val_in_limit(val=3, lim=[1, 5]))
        self.assertFalse(val_in_limit(val=6, lim=[1, 5]))
        self.assertFalse(val_in_limit(val=-1, lim=[1, 5]))
        self.assertTrue(val_in_limit(val=6, lim=[1, None]))
        self.assertFalse(val_in_limit(val=-1, lim=[1, None]))
        self.assertFalse(val_in_limit(val=6, lim=[None, 1]))
        self.assertTrue(val_in_limit(val=-1, lim=[None, 1]))

    def test_get_fizzled_composition(self):
        comp_fwids_map = get_comp_fwids_map(self.db)
        self.assertSetEqual(set(get_fizzled_composition([], self.db, comp_fwids_map)), {"SrO2"})
        self.assertSetEqual(set(get_fizzled_composition(["SrO2"], self.db, comp_fwids_map)), set())

    def test_convert(self):
        self.assertAlmostEqual(convert(10, "no_conversion"), 10)
        self.assertAlmostEqual(convert(10, "log"), 1)
        with self.assertRaises(ValueError):
            convert(10, "invalid_conversion_foo")

    def test_get_materials_properties(self):
        result = get_materials_properties(self.db,
                                          ["bandstructure_hse.bandgap", "dielectric.epsilon_avg"],
                                          [],
                                          ["no_conversion", "log"])
        actual = [3.139099999999999, math.log10(16.1041487)]
        self.assertSetEqual(set(result.keys()), {"SbO2"})
        for r, a in zip(result["SbO2"], actual):
            self.assertAlmostEqual(r, a)


if __name__ == "__main__":
    unittest.main()
