#! /usr/bin/env bash

SECONDS=0

# ── Paths ─────────────────────────────────────────────────────────────────────
ligPrep_DIR="$HOME/DeepConf"
PYTHON_DIR="$HOME/.local/Miniconda3/envs/ANI_AIMNet_NeQuIP/bin"

# ── Input ─────────────────────────────────────────────────────────────────────
struct_dir="./structures"
add_hydrogen=no

# ── Calculator ────────────────────────────────────────────────────────────────
# Available: ani1x / ani1ccx / ani2x / aimnet2 / nequip / g16 / uff
caculator_type="nequip"

# AIMNet2: set calculator_model to aimnet2, aimnet2-2025, etc. (blank = default)
# NequIP:  set calculator_model to deployed .pth path (blank = bundled G_NequIP.pth)
# G16:     calculator_charge and calculator_multiplicity are passed to Gaussian
# Leave calculator_charge blank to infer from SDF formal charge, or set an override
model_dir="$ligPrep_DIR/all_NNP_MODELS"
nequip_model_file="G_NequIP.pth"
calculator_model=""
calculator_charge="1"
calculator_multiplicity=1
calculator_device=auto
nequip_chemical_symbols=""

# ── Conformer generation ──────────────────────────────────────────────────────
genconformer=yes
ETKDG=yes
num_conformers=50
max_attempts=100000
prune_rms_thresh=0.05
nfold=2
npick=0
nscale=10

# ── Geometry optimization ─────────────────────────────────────────────────────
# optimization_method: BFGS / LBFGS / GPMin / FIRE / Berny
optimization_method=FIRE
optimization_conf=yes
optimization_lig=no
pre_optimization_lig=no
thr_fmax=0.2
maxiter=50000
opt_prune_rms_thresh=0.5
opt_prune_diffE_thresh=0.01

# ── MD sampling (alternative to ETKDG conformer generation) ──────────────────
sample_md=no
external_md_traj_file=""
run_md=no
md_temperature=400
md_steps=50000
md_timestep_fs=1.0
md_sample_interval=500
md_friction=0.01
md_box_size=20.0
md_traj_file="md_sampled_confs.xyz"

# ── Gaussian 16 (only used when caculator_type=g16) ───────────────────────────
g16_nprocs=64
g16_mem="4GB"
g16_level="WB97XD"
g16_basis="6-311++G(3df,3pd)"

# ── RMSD clustering (post-optimization) ──────────────────────────────────────
# cluster_nprocs and cluster_chunk_size default to all CPUs / auto — only set
# these to override (e.g. to limit CPU usage on a shared node)
cluster_nprocs=64
cluster_chunk_size=4000
cluster_linkage=complete

# ── Output ────────────────────────────────────────────────────────────────────
organize_clusters=yes
organize_mode=move
summary_csv=cluster_summary.csv
# verbose=yes keeps all intermediate folders; verbose=no writes only the output SDF
verbose=yes

# ── Run ───────────────────────────────────────────────────────────────────────
"$PYTHON_DIR/python" "$ligPrep_DIR/runConfGen.py" \
"$struct_dir" \
"$add_hydrogen" \
"$caculator_type" \
"$optimization_method" \
"$optimization_conf" \
"$optimization_lig" \
"$pre_optimization_lig" \
"$genconformer" \
"$g16_nprocs" \
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
"$verbose" \
"$calculator_model" \
"$calculator_charge" \
"$calculator_multiplicity" \
"$calculator_device" \
"$nequip_chemical_symbols" \
"$g16_mem" \
"$g16_level" \
"$g16_basis" \
"$sample_md" \
"$external_md_traj_file" \
"$run_md" \
"$md_temperature" \
"$md_steps" \
"$md_timestep_fs" \
"$md_sample_interval" \
"$md_friction" \
"$md_box_size" \
"$md_traj_file"


echo "Elapsed time: $SECONDS seconds"
