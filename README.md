ML-atomate
====

Machine learning-assisted Atomate code for autonomous computational materials screening.


## Installation
Download or clone the github repository, e.g., git clone https://github.com/takahashi-akira-36m/ml_atomate


## Quick start
The purpose of this code is to accelerate high-throughput calculations of Atomate code in high-performance computing (HPC) system. 

The core functions of this code are to (1) wait for new data to be obtained by first principles calculations, 
(2) construct a surrogate prediction model using machine learning techniques,
and (3) assign high priority to promising candidates using the prediction model and black-box optimization techniques.

All we have to do is to run priority_setter.sh in one of the computational nodes of the HPC system.

We will show examples, including setting up high-throughput calculations with Atomate, using the same script as in our paper.

The specific shell scripts and results are available in ml_atomate/example_stam_m_2023.


1. Prepare candidate materials.

Firstly, we should prepare candidate materials for calculation.
Such candidates can be generated by performing element substitution on a specific crystal structure type or by retrieving from a large database with specific conditions.
For example, we retrieved oxides and chalcogenides from the Materials Project.
The specific code can be run using the following command:
```shell
python ${ml_atomate_path}/ml_atomate/example_stam_m_2023/a_generate_poscars.py POSCARs
```
This will create a directory named "POSCARs" and download POSCAR files and metadata.
Those dumped POSCARs will be used in Step 3.
(Note:) In the current version, compositions must be unique because they are used as identifiers.

2. Calculate descriptors. 

Secondly, we should prepare descriptors for the candidates (i.e., numerical representation of the candidate materials) to apply machine learning. 
The descriptors must be dumped as a csv file in the following format with the name ‘descriptors.csv’, in order to be loaded into priority_setter.py.
```csv
composition,PymatgenData minimum X,PymatgenData range X,PymatgenData mean X 
Ag2O,1.93,1.51,2.433333333 
BaTiO3,0.89,2.55,2.55 
MgO,1.31,2.13,2.375
...
```
For example, we used matminer code to calculate descriptors and the specific code can be run by the following command:
```shell
python ${ml_atomate_path}/ml_atomate/example_stam_m_2023/b_calc_descriptors.py POSCARs
```
The resulting descriptors.csv file is placed in the ml_atomate/example_stam_m_2023/paper_result/ directory.

3. Setting computational workflows and builder using atomate.

Thirdly, we should set up the Atomate workflow and builder.
To make workflows, we need a YAML file containing information about 
the structures of workflows and setting for first-principles calculations, 
such as ml_atomate/example_stam_m_2023/atomate_files/diel_hybrid-gap.yaml.
To register workflows, use atwf command of atomate.
The example code can be run by the following command:
```shell
lpad reset  
zsh ${ml_atomate_path}/example_stam_m_2023/c_atomate_wf.zsh # Make workflows from POSCARs/POSCAR*  
```
We should also prepare a Builder, 
which aggregates individual computational results into a single material data.
This file will be used as the required option for the ml_atomate code in Step 4.
The example file is located in ml_atomate/example_stam_m_2023/atomate_files/run_builder.py.
Note that the filename of builder script must be named "run_builder.py" within the ml_atomate framework.

4. Use ml_atomate framework.

Finally, we should execute ml-atomate and perform high-throughput calculations.
If we use the PTR method, the command should be like this.
```shell
ml_atomate_path=(path to ml_atomate) 
python ${ml_atomate_path}/ml_atomate/priority_setter.py -df db.json -bld ${ml_atomate_path}/example_stam_m_2023/atomate_files/run_builder.py -dc descriptors.csv --objective bandstructure_hse.bandgap 4.0, dielectric.epsilon_avg 30.0, -ad -c no_conversion log
```
We note that since PHYSBO code calculate priorities based on marginal scores, which may make ML procedure time-consuming when dealing with the number of candidates is numerous.
In such case, limiting the number to be prioritized by -nws (n_write_server) option or accelerating ML procedure by -nrb (n_random_basis, details are same as the num_rand_basis of physbo.policy) option would be useful.

On the other hand, if you use BLOX, the command should be as follows:
```shell
python ${ml_atomate_path}/ml_atomate/priority_setter.py -df db.json -bld ${ml_atomate_path}/example_stam_m_2023/atomate_files/run_builder.py -dc descriptors.csv --objective bandstructure_hse.bandgap dielectric.epsilon_avg -ad -c no_conversion log --blox
```

The main required options are as follows:

- -df: Database configuration file used in the Atomate framework. See the detail in https://atomate.org/installation.html
- -bld: Builder script constructed in Step 3.
- -dc: Descriptor file constructed in Step 2.
- --blox: Use BLOX method.
- --objective: The target properties represented by the path of "materials" collection constructed by the builder. 
  If you use PTR, target ranges are also necessary ("a,b", "a," and ",b" represent "a < y < b", "a < y" and "y < b", respectively). 
- -ad: Use all descriptors written in descriptors.csv. 
  Alternatively, we can use the --n_descriptors option, 
  pruning descriptors according to the descriptor importances as determined by random forest technique.
- -c: Conversion function of target properties. If the scale of target properties largely deviated, "log" conversion may accelerate the exploration.

The description of other options is available by using the --help option of priority_setter.py.
  
Although this script can be run on the main node if the ML procedure is not computationally demanding, 
we recommend to use one child computational node.
For example, we used the example script and grid engine as following:
```shell
qsub ${ml_atomate_path}/example_stam_m_2023/d1_priority_setter_blox.zsh 
```
After a while, this script generates a logfile named priority_{date}.log.
After the log file outputs "Waiting for new acquired properties, sleeping XX sec...", 
start high-throughput calculations by using the qlaunch command.
```shell
qlaunch -r rapidfire
```
If the qlaunch command will run for a long time, the nohup command and background job may be useful.


By customizing the example scripts, you can change the candidate substances, 
first-principles calculation methods and workflows, descriptors for machine learning and target range if you use PTR.

Detailed documents for Pymatgen, Matminer and Atomate are available from the following URLs.

https://pymatgen.org

https://hackingmaterials.lbl.gov/matminer/

https://atomate.org

## Licence

This package is distributed under GNU General Public License version 3 (GPL v3) or later. 
We hope that you cite the following reference when you use ML-atomate code or data of example run 
(i.e. data placed in example_stam_m_2023/paper_result. This result is same as the data written in the following paper.):

A. Takahashi, K. Terayama, Y. Kumagai, R. Tamura and F. Oba "Fully Autonomous Materials Screening Methodology Combining First-Principles Calculations, Machine Learning and High-Performance Computing System" 
(Science and Technology of Advanced Materials: Methods, https://doi.org/10.1080/27660400.2023.2261834)

## Reference
This code is based on atomate library.  
[1] Mathew K, Montoya JH, Faghaninia A, et al., Atomate: A high-level interface to generate, execute, and analyze computational materials science workflows. Comput
Mater Sci. 2017;139:140–152.  
To use BLOX, this code employs Random Forest method implemented in scikit-learn and BLOX code implemented by Prof. Terayama. (https://github.com/tsudalab/BLOX):  
[2] Breiman L., Random Forests. Mach Learn. 2001;45:5–32.  
[3] Breiman L., Statistical Modeling: The Two Cultures (with comments and a
rejoinder by the author). Statist Sci. 2001;16(3):199–231.  
[4] Pedregosa F, Varoquaux G, Gramfort A, et al., Scikit-learn: Machine Learning in
Python. J Mach Learn Res. 2011;12:2825–2830.  
[5] Terayama K, Sumita M, Tamura R, et al., Pushing property limits in materials
discovery via boundless objective-free exploration. Chem Sci. 2020;11:5959–5968.  
To use PTR, this code employs Gaussian process implemented by PHYSBO code. (https://github.com/issp-center-dev/PHYSBO):  
[6] Motoyama Y, Tamura R, Yoshimi K et al., “Bayesian optimization package: PHYSBO” Computer Physics Communications Volume 278, September 2022, 108405.  
[7] Kishio T, Kaneko H, Funatsu K., Strategic parameter search method based on
prediction errors and data density for efficient product design, Chemom Intell Lab
Syst. 2013;127:70–79.  
[8] Tsukada Y, Takeno S, Karasuyama M, et al., Estimation of material parameters
based on precipitate shape: efficient identification of low-error region with
Gaussian process modelling. Sci Rep. 2019;9:15794.  
[9] Iwama R, Kaneko H., Design of ethylene oxide production process based on
adaptive design of experiments and Bayesian optimization. J Adv Manuf Process.
2021;3:e10085.  

And if you use example scripts, you should cite these papers:  
Poscars are retrieved by Materials Project.
[10] Jain A, Ong SP, Hautier G, et al., Commentary: The Materials Project: A materials
genome approach to accelerating materials innovation. APL Mater. 2013;1:011002.  
Descriptors are generated by matminer code
[11] Ward L, Dunn A, Faghaninia A, et al., Matminer: An open source toolkit for
materials data mining. Comput. Mater. Sci. 2018;152:60–69.  
Note that to use most of the descriptors the appropriate reference, which is implemented in Featurizer class as "citations", should be cited.

