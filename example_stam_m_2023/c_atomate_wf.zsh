#!/usr/bin/env python3

source ~/.zshrc
atwf add POSCARs/POSCAR* -l vasp -s ~/github/ml_atomate/example_stam_m_2023/atomate_files/diel_hybrid-gap.yaml -c '{"vasp_cmd": ">>vasp_cmd<<", "db_file": ">>db_file<<"}'