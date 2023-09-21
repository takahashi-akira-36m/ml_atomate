import json
import os
import unittest
import math
from pymongo import MongoClient
from monty.serialization import loadfn
from fireworks import LaunchPad


from ml_atomate.priority_setter import get_comp_fwids_map, comps_to_fwids, fwids_to_comps, val_in_limit, \
    get_fizzled_composition, convert, get_materials_properties
from ml_atomate.utils.util import parse_objective, get_from_mongo_like_str


class TestUtil(unittest.TestCase):
    def test_parse_objective(self):
        args, lim = parse_objective(["a", "b"], True)
        self.assertListEqual(args, ["a", "b"])
        self.assertListEqual(lim[0], [None, None])
        self.assertListEqual(lim[1], [None, None])
        self.assertEqual(len(lim), 2)
        args, lim = parse_objective(["a", "0,1", "b", "2,3"], False)
        self.assertListEqual(args, ["a", "b"])
        self.assertAlmostEqual(lim[0][0], 0)
        self.assertAlmostEqual(lim[0][1], 1)
        self.assertAlmostEqual(lim[1][0], 2)
        self.assertAlmostEqual(lim[1][1], 3)
        self.assertEqual(len(lim), 2)
        args, lim = parse_objective(["a", "0,", "b", ",3"], False)
        self.assertListEqual(args, ["a", "b"])
        self.assertAlmostEqual(lim[0][0], 0)
        self.assertEqual(lim[0][1], None)
        self.assertEqual(lim[1][0], None)
        self.assertAlmostEqual(lim[1][1], 3)
        self.assertEqual(len(lim), 2)

    def test_get_from_mongo_like_str(self):
        d = {"a": {"b": [1, 2]}}
        val = get_from_mongo_like_str(d, "a.b.0")
        self.assertEqual(val, 1)


if __name__ == "__main__":
    unittest.main()
