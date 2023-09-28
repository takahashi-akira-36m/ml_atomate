#!/bin/zsh
#$ -S /bin/zsh
#$ -cwd
#$ -V
#$ -j y
#$ -o priority.log
#$ -pe all_pe* 36

ml_atomate_path=${HOME}/ml_atomate
python ${ml_atomate_path}/ml_atomate/priority_setter.py -df ~/fw_config/db.json -bld ${ml_atomate_path}/example_stam_m_2023/atomate_files/run_builder.py -dc descriptors.csv --objective bandstructure_hse.bandgap 4.0, dielectric.epsilon_avg 30.0, -rs 0 -ad -nrb 0 -nws 20 -c no_conversion log

