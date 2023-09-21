import json
import os
from tqdm import tqdm
from pymatgen.core.structure import Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.core.periodic_table import Element
from monty.json import MontyEncoder


def main():
    os.mkdir("POSCARs")
    exclude_z = [1, 2]  # H and He
    exclude_z += list(i for i in range(9, 11))  # from F to Ne
    exclude_z += list(i for i in range(17, 19))  # from Cl to Ar
    exclude_z += list(i for i in range(35, 37))  # from Br to Kr
    exclude_z += list(i for i in range(53, 55))  # from I to Xe
    exclude_z += list(i for i in range(84, 86))  # from Po to Rn
    # exclude lanthanides and actinides
    exclude_z += list(i for i in range(59, 72))  # from Pr to Lu
    exclude_z += list(i for i in range(89, 104))  # from Ac to Lr
    excluded_elements = [str(Element.from_Z(z)) for z in exclude_z]
    criteria = {"elements": {"$in": ["O", "S", "Se"], "$nin": list(excluded_elements)},
                "spacegroup.number": {"$ne": 1},
                "magnetic_type": "NM",
                "e_above_hull": 0,
                "nsites": {"$lte": 40}}
    props = None
    with MPRester() as mpr:
        if props is None:
            props = list(mpr.get_doc("mp-1265").keys())
        dat = mpr.query(criteria, props)
    for d in tqdm(dat):
        mp_id = d["task_id"]
        with open(f"POSCARs/{mp_id}.json", "w") as fw:
            json.dump(d, fw, cls=MontyEncoder)
        s: Structure = d["structure"]
        s.to(fmt="POSCAR", filename=f"POSCARs/POSCAR_{mp_id}")


if __name__ == "__main__":
    main()
