from glob import glob
from multiprocessing import Pool
from pathlib import Path
import sys
from typing import List, Optional, Union
from tqdm import tqdm
import numpy as np
import pandas as pd
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.periodic_table import Element
from pymatgen.core.units import FloatWithUnit
from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.composition import (
    AtomicOrbitals,
    AtomicPackingEfficiency,
    BandCenter,
    CationProperty,
    ElectronAffinity,
    ElectronegativityDiff,
    ElementFraction,
    ElementProperty,
    IonProperty,
    Miedema,
    OxidationStates,
    Stoichiometry,
    TMetalFraction,
    ValenceOrbital,
    YangSolidSolution,
)


featurizers: List[BaseFeaturizer] = [
    AtomicOrbitals(),
    AtomicPackingEfficiency(),
    BandCenter(),
    CationProperty.from_preset("deml"),  # All error
    ElectronAffinity(),  # All error
    ElectronegativityDiff(),  # All error
    ElementFraction(),
    ElementProperty.from_preset("matminer"),
    IonProperty(),
    Miedema(),  # All error
    OxidationStates.from_preset("deml"),  # All error
    Stoichiometry(),
    TMetalFraction(),
    ValenceOrbital(),
    YangSolidSolution()
]


def val_to_float(label: str,
                 value: Optional[Union[int, float, bool, np.bool_, np.float_, np.int_, FloatWithUnit, str, Element]]):
    orbital_type_feature_labels = {"HOMO_character", "LUMO_character"}
    orbital_type_value = {"s": 0,
                          "p": 1,
                          "d": 2,
                          "f": 3,
                          "g": 4}
    element_feature_labels = {"HOMO_element", "LUMO_element"}
    if value is not None:
        if label in orbital_type_feature_labels:
            value = orbital_type_value[value]
        elif label in element_feature_labels:
            value = Element(value).Z
    return value


def calc_compositional_descriptors(poscar_path: Path):
    descriptor = dict()
    composition = Poscar.from_file(poscar_path).structure.composition
    for featurizer in featurizers:
        try:
            vals = featurizer.featurize(composition)
        except:
            vals = [None for _ in featurizer.feature_labels()]
        for label, value in zip(featurizer.feature_labels(), vals):
            descriptor[label] = val_to_float(label, value)
    return composition.reduced_formula, pd.Series(descriptor)


def main():
    poscar_dir = sys.argv[1]
    poscars = [Path(s) for s in glob(f"{poscar_dir}/POSCAR_*")]
    with Pool() as p:
        descriptors = {i: des for i, des in tqdm(p.map(calc_compositional_descriptors, poscars), total=len(poscars))}
    df = pd.DataFrame.from_dict(descriptors, orient="index").sort_index(axis=0)
    df.index.name = "composition"
    df.to_csv("descriptors.csv")


if __name__ == "__main__":
    main()

