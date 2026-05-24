#! /usr/bin/env bash


export CUDA_VISIBLE_DEVICES="0"  # one GPU usage
# export CUDA_VISIBLE_DEVICES=""  # run on CPU

# You need to set deepQM and python path for your case
PYTHON_DIR="$HOME/.local/Miniconda3/envs/ANI_AIMNet_NeQuIP/bin"
model_dir="$HOME/deepQM/all_NNP_MODELS"


ligPrep_DIR="$HOME/DeepConf"

struct_dir="./tmp_1"

# adding hydrogen if missing (yes/no) if yes, constraint heavy atoms and minimize hydrogens.
add_hydrogen=no

# set optizetions methods whichs availble in ASE (BFGS, LBFGS, GPMin, FIRE, Berny)
optimization_method=FIRE

# optimization ligand if desired before conformer generation (yes/no)
pre_optimization_lig=no

# generate conformer if desired (yes/no)
genconformer=yes

#configuration for conformer generator parameters yes or no. No uses RDKit. if ETKG yes, max_attempts and prune_rms_threshold are redundant (not used).
ETKDG=yes
num_conformers=50
max_attempts=100000

# This is RMSD threshold for generation of the conformers at the beginning used in ETKG or torsion points by rdkit
prune_rms_thresh=0.5

# this is RMSD threshold and  used for f-clustering after optimization of the picked conformers. Should be 0.1-0.5Angstrom usuually.
opt_prune_rms_thresh=0.3

# this is RMSD threshold and  used for f-clustering after optimization of the picked conformers. eV
opt_prune_diffE_thresh=0.001

# optimized-conformer RMSD clustering controls
# nprocs above is kept for the original DeepConf/G16 parameter.
# cluster_nprocs controls only post-optimization RMSD clustering.
cluster_nprocs=64
cluster_chunk_size=4000
cluster_linkage=complete
organize_clusters=yes
organize_mode=move
summary_csv=cluster_summary.csv

# verbose=yes keeps the full ligand folder with all intermediate files.
# verbose=no exports only <file_base>_output.sdf and removes the ligand folder.
verbose=no

# select caclulator type (ani2x/g16) for optimization conf
#caculator_type="g16"
caculator_type="aimnet2"

local_model_path="$model_dir/G_NequIP.pth"


# perform geometry optimization for conformers if desired (yes/no)
optimization_conf=yes

# perform geometry optimization for orginal ligand if desired (yes/no)
optimization_lig=no

# set number of procssors for g16 calcultor (default=all cpu)
nprocs=1

# set thrshold fmax for optimization (default=0.01)
thr_fmax=0.2

#maximum iteration for optimization
maxiter=50000

# number of fold for conformer generation
nfold=3
# to pick randomly extra conformer
npick=0

nscale=10



$PYTHON_DIR/python "$ligPrep_DIR/runConfGen.py" \
"$struct_dir" \
"$add_hydrogen" \
"$caculator_type" \
"$local_model_path" \
"$optimization_method" \
"$optimization_conf" \
"$optimization_lig" \
"$pre_optimization_lig" \
"$genconformer" \
"$nprocs" \
"$thr_fmax" \
"$maxiter" \
"$ETKDG" \
"$num_conformers" \
"$max_attempts" \
"$prune_rms_thresh" \
"$opt_prune_rms_thresh" \
"$opt_prune_diffE_thresh" \
"$nfold" \
"$npick" \
"$nscale" \
"$cluster_nprocs" \
"$cluster_chunk_size" \
"$cluster_linkage" \
"$organize_clusters" \
"$organize_mode" \
"$summary_csv" \
"$verbose"



