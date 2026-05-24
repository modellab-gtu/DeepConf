#! /usr/bin/env bash

ligPrep_DIR="$HOME/Desktop/DeepConf"
PYTHON_DIR="$HOME/miniconda3/bin"

# Set this to your ligand/input directory.
struct_dir="./test_akocak"

# adding hydrogen if missing (yes/no) if yes, constrain heavy atoms and minimize hydrogens.
add_hydrogen=no

# set optimization method available in ASE (BFGS, LBFGS, GPMin, FIRE, Berny)
optimization_method=BFGS

# optimize ligand before conformer generation (yes/no)
pre_optimization_lig=yes

# generate conformers (yes/no)
genconformer=yes

# conformer generator parameters. If ETKDG=yes, max_attempts and prune_rms_thresh are not used.
ETKDG=yes
num_conformers=1000
max_attempts=100000

# RMSD threshold for initial RDKit conformer pruning.
prune_rms_thresh=0.0005

# RMSD threshold for post-optimization conformer clustering.
opt_prune_rms_thresh=0.5

# Energy threshold for post-RMSD representative pruning.
opt_prune_diffE_thresh=0.01

# calculator type (ani2x/g16/uff)
caculator_type="ani2x"

# optimize generated conformers (yes/no)
optimization_conf=yes

# optimize original ligand only (yes/no)
optimization_lig=no

# number of processors for Gaussian and RMSD clustering defaults
nprocs=1

# optimization force threshold
thr_fmax=0.002

# maximum optimization iterations
maxiter=50000

# conformer sampling controls
nfold=2
npick=0
nscale=10

# optimized-conformer RMSD clustering controls
cluster_nprocs=1
cluster_chunk_size=4000
cluster_linkage=complete
organize_clusters=yes
organize_mode=move
summary_csv=cluster_summary.csv

# verbose=yes keeps all folders/intermediates.
# verbose=no exports <file_base>_output.sdf to this directory and removes the ligand work folder.
verbose=yes

"$PYTHON_DIR/python" "$ligPrep_DIR/runConfGen.py" \
"$struct_dir" \
"$add_hydrogen" \
"$caculator_type" \
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

