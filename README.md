# DeepConf

DeepConf explores low-energy conformations of small molecules using RDKit conformer generation and machine-learning or quantum-mechanical calculators for energy evaluation and geometry optimization. It can be used with ANI-family neural network potentials, AIMNet2, in-house NequIP-ML models, and optionally Gaussian16. It is designed to generate representative conformers for downstream workflows such as docking and molecular dynamics.

## Citation

> **DeepConf: Leveraging ANI-ML Potentials for Exploring Local Minima with Application to Bioactive Conformations**  
> Omer Tayfuroglu, Irem Nur Zengin, Mehmet Serdar Koca, Abdulkadir Kocak  
> *Journal of Chemical Information and Modeling*, 2024  
> DOI: [10.1021/acs.jcim.4c02053](https://doi.org/10.1021/acs.jcim.4c02053)

> **Data-Efficient Equivariant NNPs Enable DFT-Accurate Simulations and Implicit Solvation Free Energies**  
> Esma Mutlu, Selonou G. Kankinou, Omer Tayfuroglu, Abdulkadir Kocak  
> *The Journal of Physical Chemistry B*, 2025  
> DOI: [10.1021/acs.jpcb.5c05891](https://www.doi.org/10.1021/acs.jpcb.5c05891)

## Requirements

- Conda with Python 3.11 or newer recommended for the combined ANI/AIMNet2/NequIP environment
- [RDKit](https://www.rdkit.org/)
- [Open Babel](https://openbabel.org/) Python bindings
- [ASE](https://wiki.fysik.dtu.dk/ase/)
- [TorchANI](https://aiqm.github.io/torchani/) or the corresponding ANI-family calculator environment
- [AIMNet2](https://github.com/isayevlab/aimnetcentral/tree/main) for AIMNet2 calculator support
- PyTorch
- NequIP-ML model files and calculator setup for in-house NequIP workflows
- Optional: [Gaussian16](https://gaussian.com/g16/) for QM-level optimization

Using a GPU is recommended for ANI, AIMNet2, and NequIP-ML calculations.

## Installation

```bash
git clone https://github.com/modellab-gtu/DeepConf.git
cd DeepConf

conda create --name ANI_AIMNet_NeQuIP python=3.11
conda activate ANI_AIMNet_NeQuIP
```

Repository: [modellab-gtu/DeepConf](https://github.com/modellab-gtu/DeepConf)

Install the core scientific Python dependencies:

```bash
conda install -c conda-forge numpy pandas tqdm rdkit openbabel ase pytorch torchani dftd3-python auto3d
conda install -c psi4 dftd3
```

Install AIMNet2. For CPU or the PyTorch-default install:

```bash
pip install aimnet
```

For CUDA, install a matching PyTorch build first, then AIMNet2. For example, for CUDA 12.4:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install aimnet
```

Install NequIP support for NequIP-ML workflows:

```bash
pip install nequip==0.6.1
```

For AIMNet2 details, see [aimnetcentral](https://github.com/isayevlab/aimnetcentral/tree/main).

The in-house NequIP `.pth` model files are bundled under `all_NNP_MODELS/`, so they are available after cloning the repository.

For Gaussian16 support, make sure the `g16` command is available in your shell environment.

If the installation commands above do not reproduce the working environment on your machine, compare against [`pip-list-ANI_AIMNet_NeQuIP.txt`](pip-list-ANI_AIMNet_NeQuIP.txt). It records a known working `pip list` for troubleshooting, but it is not intended to be a strict lock file.

### Clean WSL Installation

For Windows users, use WSL2 with a Linux distribution such as Ubuntu. Keep the repository and run folders inside the WSL Linux filesystem, for example under `$HOME/projects`, instead of running from a Windows path such as `/mnt/c/Users/.../Documents/GitHub`. This avoids slow file I/O and path issues.

Start from a clean WSL shell:

```bash
sudo apt update
sudo apt install -y git wget ca-certificates bzip2
mkdir -p "$HOME/projects"
cd "$HOME/projects"
git clone https://github.com/modellab-gtu/DeepConf.git
cd DeepConf
```

If Conda is not already available in WSL, install Miniconda for Linux first and restart the shell, or source Conda manually:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
```

Create a fresh DeepConf environment and install the core dependencies:

```bash
conda create -y --name ANI_AIMNet_NeQuIP python=3.11
conda activate ANI_AIMNet_NeQuIP
conda install -y -c conda-forge numpy pandas tqdm rdkit openbabel ase pytorch torchani dftd3-python auto3d
```

The additional `psi4` `dftd3` package is optional for the ANI/AIMNet2/NequIP workflows:

```bash
conda install -y -c psi4 dftd3
```

Install AIMNet2 and NequIP support when those calculators are needed:

```bash
python -m pip install aimnet nequip==0.6.1
```

Check the environment:

```bash
python -c "import numpy, pandas, rdkit, ase, openbabel, torch, torchani; print('DeepConf core imports OK')"
python -m py_compile runConfGen.py core_conf.py
```

When using `runConfGen.sh` from WSL, set the repository and Python paths to the WSL install locations:

```bash
ligPrep_DIR="$HOME/projects/DeepConf"
PYTHON_DIR="$HOME/miniconda3/envs/ANI_AIMNet_NeQuIP/bin"
```

For GPU use in WSL, first confirm that Windows NVIDIA drivers and WSL GPU passthrough are working:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If `torch.cuda.is_available()` is `False`, DeepConf can still run CPU-side for small tests, but ML-calculator production runs will be much slower. If `pip install aimnet` replaces PyTorch with a CUDA build that does not match your driver, install a PyTorch build matching your WSL/NVIDIA driver before installing AIMNet2, or use the CPU-side environment for smoke tests.

## Quick Start

Create a work directory and put your ligand files in your own input folder. The repository does not ship with a default `test/` input folder.

```bash
mkdir deepconf_run
cd deepconf_run
mkdir structures
cp /path/to/your_ligands/*.sdf structures/
```

Copy the run script and edit it for your machine:

```bash
cp /path/to/DeepConf/runConfGen.sh .
```

Open `runConfGen.sh` and set at least these values:

```bash
ligPrep_DIR="/path/to/DeepConf"
PYTHON_DIR="/path/to/conda/envs/DeepConf/bin"
struct_dir="./structures"
```

Then run:

```bash
bash runConfGen.sh
```

## Main Options In `runConfGen.sh`

`runConfGen.sh` is the recommended user-facing entry point. It passes values positionally to `runConfGen.py`, so keep the order at the bottom of the script unchanged when editing.

### Paths And Input

| Option | Current default | Description |
| --- | --- | --- |
| `ligPrep_DIR` | `$HOME/DeepConf` | Absolute path to the DeepConf repository. The script runs `$ligPrep_DIR/runConfGen.py` and uses `$ligPrep_DIR/all_NNP_MODELS` for bundled NequIP models. |
| `PYTHON_DIR` | `$HOME/.local/Miniconda3/envs/ANI_AIMNet_NeQuIP/bin` | Directory containing the Python executable for the DeepConf environment. The script runs `$PYTHON_DIR/python`. |
| `struct_dir` | `./test` | Directory containing input ligand files. Each file in this folder is processed. Hidden files are ignored. |
| `verbose` | `yes` | `yes` keeps all folders and intermediate files. `no` copies the final SDF to the run directory as `<file_base>_output.sdf` and removes each ligand work folder. |

### Workflow Switches

| Option | Current default | Description |
| --- | --- | --- |
| `add_hydrogen` | `no` | `yes` uses Open Babel to add missing hydrogens. Added H atoms are then relaxed with heavy atoms fixed using the selected calculator. |
| `pre_optimization_lig` | `no` | `yes` optimizes the input ligand before conformer generation. Use this for a pre-relaxed starting geometry. |
| `genconformer` | `yes` | `yes` generates conformers with RDKit. `no` skips conformer generation. |
| `sample_md` | `no` | `yes` uses frames from `external_md_traj_file` as the conformer pool instead of generating RDKit conformers. The input ligand file is still used as the topology template. |
| `external_md_traj_file` | blank | External MD trajectory file used when `sample_md=yes`. SDF trajectories are read with RDKit; other formats such as XYZ/PDB are read with ASE. Relative paths are checked from the run directory first, then from `struct_dir`. |
| `run_md` | `no` | `yes` runs internal ASE Langevin MD with the selected calculator, writes `md_traj_file`, and then uses those frames as the conformer pool. |
| `optimization_conf` | `yes` | `yes` optimizes the picked/generated conformers with the selected calculator. |
| `optimization_lig` | `no` | `yes` optimizes only the original ligand when `genconformer=no`. |

Common workflows:

| Goal | Settings |
| --- | --- |
| Ligand-only geometry optimization | `genconformer=no`, `optimization_lig=yes`, `optimization_conf=no`, `pre_optimization_lig=no` |
| Conformer generation without pre-optimization | `genconformer=yes`, `pre_optimization_lig=no` |
| Pre-optimization plus conformer generation | `genconformer=yes`, `pre_optimization_lig=yes` |
| External MD trajectory sampling | `sample_md=yes`, `external_md_traj_file=<trajectory>`. `genconformer` can stay `yes`; RDKit embedding is skipped when `sample_md=yes`. |
| Internal ASE MD sampling | `run_md=yes`. DeepConf runs MD with the selected calculator, writes an XYZ trajectory, and then follows the same MD-frame sampling path. |
| Generate conformers but do not optimize picked conformers | `genconformer=yes`, `optimization_conf=no` |
| Generate and optimize picked conformers | `genconformer=yes`, `optimization_conf=yes` |

### Calculator Settings

| Option | Current default | Description |
| --- | --- | --- |
| `caculator_type` | `aimnet2` | Calculator choice. The variable name is intentionally kept as `caculator_type` for backward compatibility. Supported values include `ani1x`, `ani1ccx`, `ani2x`, `aimnet2`, `nequip`, `g16`, and `uff`. |
| `calculator_model` | blank | Optional model name or model path. For AIMNet2, leave blank to use `aimnet2` or set a specific AIMNet model name. For NequIP, leave blank to use the bundled `all_NNP_MODELS/G_NequIP.pth`, or set this to a deployed `.pth` model path or a newer compiled model path. |
| `calculator_charge` | blank | AIMNet2 and Gaussian16 total molecular charge. Leave blank to infer the formal charge from the loaded SDF/RDKit molecule. Set an integer such as `-1`, `0`, or `1` to override. |
| `calculator_multiplicity` | `1` | AIMNet2 and Gaussian16 spin multiplicity. |
| `calculator_device` | `auto` | Device for NequIP. `auto` selects CUDA when PyTorch sees a GPU, otherwise CPU. You can force `cpu` or `cuda`. |
| `nequip_chemical_symbols` | blank | Optional NequIP species mapping. For bundled NequIP models, blank automatically uses identity mapping. Use `identity` for other models whose atom-type names match chemical symbols, or provide a comma-separated list/JSON mapping if required by the model. |
| `g16_mem` | `4GB` | Gaussian16 memory setting passed to ASE Gaussian as `mem`. |
| `g16_level` | `WB97XD` | Gaussian16 functional/method setting passed to ASE Gaussian as `xc`. |
| `g16_basis` | `6-311++G(3df,3pd)` | Gaussian16 basis-set setting passed to ASE Gaussian as `basis`. |
| `nprocs` | `64` | Number of processors for Gaussian jobs. It does not make ANI, AIMNet2, or NequIP single-structure optimization run on 64 CPU cores. Use GPU testing for ML-calculator speed evaluation. |

Calculator notes:

- ANI-family choices use TorchANI: `ani1x`, `ani1ccx`, or `ani2x`.
- AIMNet2 uses the AIMNet ASE calculator. Charge is inferred from formal charges in the SDF unless `calculator_charge` or `DEEPCONF_AIMNET_CHARGE` is set.
- NequIP uses the ASE `NequIPCalculator`. Older `.pth` or `.pt` models are loaded through the deployed-model path. Raw NequIP cohesive energies are converted back to total eV energies before writing SDF/CSV `Energy` values.
- `g16` uses Gaussian16 through ASE and uses `nprocs` for `nprocshared`. Charge is inferred from formal charges in the SDF unless `calculator_charge` or `DEEPCONF_G16_CHARGE` is set, `calculator_multiplicity` is passed as the Gaussian multiplicity, and `g16_mem`, `g16_level`, and `g16_basis` control the Gaussian memory, method, and basis settings.
- `uff` is only for RDKit/MM energy handling in supported paths. Do not use `uff` with `optimization_conf=yes`.

### Bundled NequIP Models

The repository includes in-house NequIP model files under:

```bash
all_NNP_MODELS/
```

`runConfGen.sh` defines:

```bash
model_dir="$ligPrep_DIR/all_NNP_MODELS"
nequip_model_file="G_NequIP.pth"
```

Bundled model files:

| File | Description |
| --- | --- |
| `G_NequIP.pth` | Default bundled NequIP model used by the run script template. |
| `G_NequIP_smdW.pth` | Bundled NequIP model variant. |
| `M_NequIP.pth` | Bundled NequIP model variant. |
| `M_NequIP_smdO.pth` | Bundled NequIP model variant. |
| `M_NequIP_smdW.pth` | Bundled NequIP model variant. |

To use the default bundled NequIP model, edit `runConfGen.sh` like this:

```bash
caculator_type="nequip"
calculator_device=auto
```

When `caculator_type="nequip"` and `calculator_model` is blank, DeepConf uses `all_NNP_MODELS/G_NequIP.pth` automatically and applies identity species mapping for the bundled model. To use another bundled model, set `nequip_model_file` and `calculator_model`, for example:

```bash
nequip_model_file="M_NequIP_smdW.pth"
calculator_model="$model_dir/$nequip_model_file"
```

The `all_NNP_MODELS/yml_file/` folder contains reference environment YAML files for the model-generation environments. They are included for provenance and troubleshooting, not required for normal DeepConf runs.

### Optimization Settings

| Option | Current default | Description |
| --- | --- | --- |
| `optimization_method` | `FIRE` | ASE optimizer for geometry optimization. Common choices are `BFGS`, `LBFGS`, `FIRE`, `GPMin`, `Berny`, `CG`, and `NewtonRaphson` where available. |
| `thr_fmax` | `0.2` | Force convergence threshold passed to ASE optimizers as `fmax`. Lower values are stricter and slower. |
| `maxiter` | `50000` | Maximum geometry-optimization steps. |

### Internal ASE MD Sampling

| Option | Current default | Description |
| --- | --- | --- |
| `run_md` | `no` | `yes` runs ASE Langevin MD before conformer sampling. No prior minimization is performed unless `pre_optimization_lig=yes` is also set. |
| `md_temperature` | `400` | MD temperature in K. |
| `md_steps` | `50000` | Number of MD steps. Total simulation time is `md_steps * md_timestep_fs`. |
| `md_timestep_fs` | `1.0` | MD time step in fs. |
| `md_sample_interval` | `500` | Write one trajectory frame every this many MD steps. For example, `100` saves frames at steps 0, 100, 200, etc. |
| `md_friction` | `0.01` | ASE Langevin friction in inverse fs. |
| `md_box_size` | `20.0` | Cubic cell length in Angstrom. The cell is non-periodic and used only to hold the molecule. |
| `md_traj_file` | `md_sampled_confs.xyz` | Extended XYZ trajectory written under each ligand work folder and then used as the conformer source. Each frame header includes box vectors, MD step, time, total energy, potential energy, kinetic energy, and instantaneous temperature. Atom lines include coordinates and velocities. |

Internal MD currently uses the selected ASE-compatible calculator, such as ANI, AIMNet2, or NequIP. Classical OpenMM/GROMACS/AMBER-style workflows are not part of this first internal MD implementation.

### RDKit Conformer Generation

| Option | Current default | Description |
| --- | --- | --- |
| `ETKDG` | `yes` | `yes` uses RDKit ETKDGv3 embedding. When `ETKDG=yes`, `max_attempts` and `prune_rms_thresh` are not used by the ETKDG branch. |
| `num_conformers` | `50` | Requested number of RDKit conformers. DeepConf may increase this internally based on torsion count, `nfold`, and `nscale`. |
| `max_attempts` | `100000` | Maximum embedding attempts for the non-ETKDG RDKit embedding path. |
| `prune_rms_thresh` | `0.05` | Initial RDKit conformer pruning RMSD threshold for the non-ETKDG embedding path. Smaller values keep more similar conformers. |
| `nfold` | `2` | Torsion-count scaling factor used when DeepConf estimates a minimum conformer count. |
| `npick` | `0` | Number of random conformers picked from each initial cluster in addition to the minimum-energy conformer. `0` keeps only the minimum-energy conformer from each initial cluster. |
| `nscale` | `10` | Additional scaling factor for the number of generated conformers based on torsion count. |

When `sample_md=yes`, this RDKit embedding step is skipped. DeepConf loads each frame from `external_md_traj_file` as a conformer using the input molecule's RDKit topology, calculates energies, writes `confs_cluster_*` SDF files with `Energy` properties, and then continues through the same picked-conformer, optional optimization, RMSD clustering, and final-output pipeline.

### RMSD Clustering And Final Output

| Option | Current default | Description |
| --- | --- | --- |
| `opt_prune_rms_thresh` | `0.5` | RMSD threshold used for post-optimization conformer clustering. This is an actual RMSD threshold in Angstrom. |
| `opt_prune_diffE_thresh` | `0.01` | Energy threshold for pruning cluster representatives after RMSD clustering. |
| `cluster_nprocs` | `64` | Number of processes used for post-optimization RMSD matrix calculation and clustering. This can use many CPU cores during RMSD comparisons. |
| `cluster_chunk_size` | `4000` | Multiprocessing chunk size for RMSD pair calculations. Larger chunks reduce overhead; smaller chunks may help load balancing for very uneven cases. |
| `cluster_linkage` | `complete` | Linkage method for hierarchical RMSD clustering. `complete` is the recommended/default mode. |
| `organize_clusters` | `yes` | `yes` creates energy-ranked `cluster_1`, `cluster_2`, ... directories. |
| `organize_mode` | `move` | `move` places conformer files into cluster directories. `copy` keeps originals and copies them into cluster directories. |
| `summary_csv` | `cluster_summary.csv` | Name of the cluster summary CSV written under `opt_picked_confs/`. |

The positional command-line interface is backward-compatible: clustering, `verbose`, and calculator-specific options are trailing arguments.

NequIP model settings can alternatively be supplied with environment variables:

```bash
export DEEPCONF_NEQUIP_MODEL=/path/to/compiled_or_deployed_model
export DEEPCONF_NEQUIP_CHEMICAL_SYMBOLS=identity
```

For AIMNet2, `calculator_model` can be left blank to use `aimnet2`, or set with:

```bash
export DEEPCONF_AIMNET_MODEL=aimnet2
```

The AIMNet2 charge is inferred from the molecule's formal charge by default. To override it explicitly:

```bash
export DEEPCONF_AIMNET_CHARGE=-1
```

For Gaussian16, the same `calculator_charge` setting is used, or it can be overridden with:

```bash
export DEEPCONF_G16_CHARGE=1
```

## Output

Each input ligand is processed in its own work folder named after the input file base name.

For conformer generation with conformer optimization:

```text
<file_base>/
  opt_picked_confs/
    <file_base>_output.sdf
    cluster_summary.csv
    cluster_1/
    cluster_2/
    ...
```

`<file_base>_output.sdf` contains the final representative conformers sorted by energy. Cluster directories are energy-ranked: `cluster_1` contains the cluster whose representative has the lowest energy.

For conformer generation without conformer optimization:

```text
<file_base>/
  picked_confs/
    <file_base>_output.sdf
```

This SDF contains the picked conformers sorted by energy.

For compact output mode:

```bash
verbose=no
```

DeepConf copies the final SDF to the run directory:

```text
<file_base>_output.sdf
```

and removes the ligand work folder.

For ligand-only optimization, outputs are written as:

```text
<file_base>/
  global_<prefix><file_base>.sdf
  global_<prefix><file_base>_energy.txt
```

Timing and failure logs are written in the run directory:

```text
timings.csv
failed_files.csv
```

## Notes

- RMSD clustering uses direct RDKit `GetBestRMS` comparisons with complete linkage by default.
- Optimized conformer SDF files preserve the RDKit topology and update only coordinates from ASE. This keeps RMSD clustering consistent while preserving the optimized geometry.
- Do not use `uff` with `optimization_conf=yes`; the script exits because UFF conformer optimization is not supported in that path.
