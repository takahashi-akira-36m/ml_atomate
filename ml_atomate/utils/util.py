from math import isnan
from numbers import Real
from typing import List, Iterator, Union, Any


def parse_objective(arg: List[str], mode_blox=False):
    if mode_blox:
        return arg, [[None, None] for _ in range(len(arg))]
    if len(arg) % 2 != 0:
        raise ValueError(f"args must be even, but {len(arg)}")
    it: Iterator[str] = iter(arg)
    objectives = []
    limits = []
    for obj, lim in zip(it, it):
        objectives.append(obj)
        if len([s for s in [",", ">", "<"] if s in lim]) != 1:
            raise ValueError("ONE of ~, >, or < must be specified.")
        elif "," in lim:
            if lim.startswith(","):
                low = None
                high = float(lim.split(",")[1])
            elif lim.endswith(","):
                low = float(lim.split(",")[0])
                high = None
            else:
                low, high = (float(v) for v in lim.split(","))
        elif ">" in lim:
            if lim.startswith(">"):
                high = None
                low = float(lim.split(">")[1])
            elif lim.endswith(">"):
                high = float(lim.split(">")[0])
                low = None
            else:
                high, low = (float(v) for v in lim.split(">"))
        elif "<" in lim:
            if lim.startswith("<"):
                low = None
                high = float(lim.split("<")[1])
            elif lim.endswith("<"):
                low = float(lim.split("<")[0])
                high = None
            else:
                low, high = (float(v) for v in lim.split("<"))
        limits.append((low, high))
    return objectives, limits


def get_from_mongo_like_str(d: Union[dict, list], s: str) -> Any:
    if isinstance(d, Real) and isnan(d):
        return float("nan")
    k, sep, s2 = s.partition(".")
    if isinstance(d, list):
        k = int(k)
    if d == "Not found in task":
        return float("nan")
    try:
        if sep:
            return get_from_mongo_like_str(d[k], s2)
        else:
            return d[k]
    except (KeyError, IndexError):
        return float("nan")
