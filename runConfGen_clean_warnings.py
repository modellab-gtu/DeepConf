
from rdkit import Chem
from rdkit.Chem import rdMolAlign
import numpy as np

from ase.io import read, write

# Hide the requests/chardet warning when the environment lacks optional charset packages.
# The better environment-level fix is to install charset-normalizer or chardet.
import warnings
warnings.filterwarnings(
    "ignore",
    message="Unable to find acceptable character detection dependency.*"
)

from core_conf import confGen
import argparse
import os, sys, shutil
import multiprocessing
from itertools import product
import time


nprocs_all = int(multiprocessing.cpu_count())



parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("structure_dir", type=str)
parser.add_argument("ignore_hydrogen", nargs="?", default="No") # args for bool
parser.add_argument("calculator_type", type=str)
parser.add_argument("local_model_path", type=str)
parser.add_argument("optimization_method", nargs="?", default="No") # args for bool
parser.add_argument("optimization_conf", nargs="?", default="No") # args for bool
parser.add_argument("optimization_lig", nargs="?", default="No") # args for bool
parser.add_argument("pre_optimization_lig", nargs="?", default="No") # args for bool
parser.add_argument("genconformer", nargs="?", default="No") # args for bool
parser.add_argument("nprocs", type=int, default=nprocs_all)
parser.add_argument("thr_fmax", type=float, default=0.05)
parser.add_argument("maxiter", type=int, default=500)

parser.add_argument("ETKDG", nargs="?", default="No") # args for bool
parser.add_argument("num_conformers", type=int, default=50)
parser.add_argument("max_attempts", type=int, default=100)
parser.add_argument("prune_rms_thresh", type=float, default=0.2)
parser.add_argument("opt_prune_rms_thresh", type=float, default=0.2)
parser.add_argument("opt_prune_diffE_thresh", type=float, default=0.001)
parser.add_argument("nfold", type=int, default=2)
parser.add_argument("npick", type=int, default=2)
parser.add_argument("nscale", type=int, default=2)

# Optional optimized-conformer RMSD clustering controls.
# They are trailing positional arguments so older runConfGen_1.sh calls remain compatible.
parser.add_argument("cluster_nprocs", nargs="?", type=int, default=nprocs_all)
parser.add_argument("cluster_chunk_size", nargs="?", type=int, default=4000)
parser.add_argument("cluster_linkage", nargs="?", default="complete")
parser.add_argument("organize_clusters", nargs="?", default="yes")
parser.add_argument("organize_mode", nargs="?", default="move")
parser.add_argument("summary_csv", nargs="?", default="cluster_summary.csv")
# If verbose=yes, keep the full ligand work directory.
# If verbose=no, export only the final SDF to the parent run directory and remove the work directory.
parser.add_argument("verbose", nargs="?", default="yes")


def calcFuncRunTime(func):
    import time
    def wrapper(*args, **kwargs):
        s_time = time.time()
        func(*args, **kwargs)
        print(f"Funtion {func.__name__} executed in {(time.time()-s_time)/60:.4f} m")
    return wrapper


def getBoolStr(string):
    string = string.lower()
    if "true" in string or "yes" in string:
        return True
    elif "false" in string or "no" in string:
        return False
    else:
        print("%s is bad input!!! Must be Yes/No or True/False" %string)
        sys.exit(1)


def _read_sdf_energy_for_export(sdf_path, default=float("inf")):
    """Read Energy from the first molecule in an SDF, for fallback ranking only."""
    try:
        mol = next(Chem.SDMolSupplier(sdf_path, removeHs=False))
    except Exception:
        return default

    if mol is None:
        return default

    if mol.HasProp("Energy"):
        try:
            return float(mol.GetProp("Energy"))
        except Exception:
            return default

    return default


def _write_combined_sdf_from_dir(sdf_dir, out_sdf):
    """
    Fallback final-output creator.
    Used only if a directory contains individual SDFs but no *_output.sdf yet.
    Molecules are written in increasing Energy order when Energy is available.
    """
    sdf_files = [
        os.path.join(sdf_dir, f)
        for f in os.listdir(sdf_dir)
        if f.endswith(".sdf")
    ]

    if len(sdf_files) == 0:
        return None

    sdf_files = sorted(sdf_files, key=lambda p: _read_sdf_energy_for_export(p))

    with Chem.SDWriter(out_sdf) as writer:
        for rank, sdf_path in enumerate(sdf_files, start=1):
            try:
                mol = next(Chem.SDMolSupplier(sdf_path, removeHs=False))
            except Exception:
                mol = None

            if mol is None:
                continue

            mol.SetProp("SourceFile", os.path.basename(sdf_path))
            mol.SetProp("OutputRankByEnergy", str(rank))
            writer.write(mol)

    if os.path.exists(out_sdf) and os.path.getsize(out_sdf) > 0:
        return out_sdf

    return None


def _find_final_sdf(WORK_DIR, file_base, prefix):
    """
    Locate the final SDF regardless of the workflow branch.

    Priority:
    1) DeepConf output SDFs from optimized conformer pruning, usually:
       WORK_DIR/opt_picked_confs/<file_base>_output.sdf
       or, for a single conformer:
       WORK_DIR/opt_picked_confs/opt_output.sdf
    2) Any *_output.sdf under WORK_DIR.
    3) Ligand optimization output:
       WORK_DIR/global_<prefix><file_base>.sdf
    4) Pre-optimization-only output:
       WORK_DIR/pre_<prefix><file_base>.sdf
    5) If genconformer=yes but optimization_conf=no and no output SDF exists yet,
       combine individual picked conformers into WORK_DIR/<file_base>_output.sdf.
    """
    candidates = []

    exact_names = [
        f"{file_base}_output.sdf",
        f"{prefix}output.sdf",
        "output.sdf",
    ]

    for root, dirs, files in os.walk(WORK_DIR):
        for name in files:
            if name in exact_names and name.endswith(".sdf"):
                candidates.append(os.path.join(root, name))

    if candidates:
        # Prefer deeper picked_confs/opt_picked_confs outputs over accidental top-level files.
        candidates = sorted(
            candidates,
            key=lambda p: (
                0 if "picked_confs" in os.path.normpath(p).split(os.sep) else 1,
                len(os.path.normpath(p).split(os.sep)),
                p,
            )
        )
        return candidates[0]

    # Any *_output.sdf under WORK_DIR.
    output_candidates = []
    for root, dirs, files in os.walk(WORK_DIR):
        for name in files:
            if name.endswith("_output.sdf"):
                output_candidates.append(os.path.join(root, name))

    if output_candidates:
        output_candidates = sorted(
            output_candidates,
            key=lambda p: (
                0 if "picked_confs" in os.path.normpath(p).split(os.sep) else 1,
                len(os.path.normpath(p).split(os.sep)),
                p,
            )
        )
        return output_candidates[0]

    # Direct ligand optimization branch.
    global_candidate = os.path.join(WORK_DIR, f"global_{prefix}{file_base}.sdf")
    if os.path.exists(global_candidate):
        return global_candidate

    # Pre-optimization-only branch.
    pre_candidate = os.path.join(WORK_DIR, f"pre_{prefix}{file_base}.sdf")
    if os.path.exists(pre_candidate):
        return pre_candidate

    # Fallback for genconformer=yes, optimization_conf=no.
    for subdir_name in ("picked_confs", "opt_picked_confs"):
        sdf_dir = os.path.join(WORK_DIR, subdir_name)
        if os.path.isdir(sdf_dir):
            combined = os.path.join(WORK_DIR, f"{file_base}_output.sdf")
            made = _write_combined_sdf_from_dir(sdf_dir, combined)
            if made is not None:
                return made

    return None


def _export_final_sdf_and_cleanup(WORK_DIR, file_base, prefix):
    """
    For compact mode: copy final SDF one level above WORK_DIR and then remove WORK_DIR.
    If no final SDF is found, keep WORK_DIR to avoid data loss.
    """
    final_sdf = _find_final_sdf(WORK_DIR, file_base, prefix)

    if final_sdf is None or not os.path.exists(final_sdf):
        print(f"Warning: no final SDF found for {file_base}; keeping work directory: {WORK_DIR}")
        return None

    export_sdf = os.path.abspath(f"{file_base}_output.sdf")

    if os.path.exists(export_sdf):
        os.remove(export_sdf)

    shutil.copy2(final_sdf, export_sdf)
    print(f"Compact output written to: {export_sdf}")

    shutil.rmtree(WORK_DIR)
    print(f"Removed work directory because verbose=no: {WORK_DIR}")

    return export_sdf


def setG16calculator(lig, file_base, label, WORK_DIR):
    lig.setG16Calculator(
            label="%s/g16_%s/%s"%(WORK_DIR, label, file_base),
            mem="20GB",
            chk="",
            nprocs=nprocs,
            xc="WB97XD",
            basis="6-311++G(3df,3pd)",
            charge=1,
            scf="XQC, maxconventionalcycles=100",
            extra="nosymm",

            )
    return lig


def setGenConformers(lig, out_file_path, mmCalculator):
    trial = lig.n_trial
    while trial <= 3:
        try:
            lig.genGonformers(
                file_path=out_file_path,
                numConfs=num_conformers,
                ETKDG=ETKDG,
                maxAttempts=max_attempts,
                pruneRmsThresh=prune_rms_thresh,
                mmCalculator=mmCalculator,
                optimization_conf=optimization_conf,
                opt_prune_rms_thresh=opt_prune_rms_thresh,
                opt_prune_diffE_thresh=opt_prune_diffE_thresh,
                nfold=nfold,
                npick=npick,
                cluster_nprocs=cluster_nprocs,
                cluster_chunk_size=cluster_chunk_size,
                cluster_linkage=cluster_linkage,
                organize_clusters=organize_clusters,
                organize_mode=organize_mode,
                summary_csv=summary_csv,)
        except:
            print(f"Trail {trial} failed, attempting new one... ")
            lig.increaseTrilNum()
            trial = lig.n_trial
            setGenConformers(lig, out_file_path, mmCalculator)
        finally:
            return lig
    else:
        print(f"{trial -1} attempts failed, Skipping...")
        return None


#  @calcFuncRunTime
def runConfGen(file_name):
    "Starting ligand preparetion process... "
    mol_path= "%s/%s"%(structure_dir, file_name)

    file_base = file_name.split(".")[0]
    #create destination directory
    WORK_DIR = file_base
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.mkdir(WORK_DIR)

    #Flags
    # default mm calculator set to False
    mmCalculator=False
    # default adding H is False
    addH = False

    # if desire adding H by openbabel
    prefix = ""
    if ignore_hydrogen:
        addH = True
        prefix += "addH_"
    if optimization_lig or optimization_conf or pre_optimization_lig:
        prefix += "opt_"

    # initialize confGen
    lig = confGen(mol_path, addH, WORK_DIR)
    lig.setVerbose(verbose)
    lig.setOptMethod(optimization_method)
    #  lig.writeRWMol2File("test/test.xyz")

    if "ani2x" in calculator_type.lower():
        lig.setANI2XCalculator()
    if "aimnet2" in calculator_type.lower():
        lig.setAIMNet2alculator()
    if "nequip" in calculator_type.lower():
        lig.setNequIPCalculator(nequip_model_path)
    elif "g16" in calculator_type.lower():
        lig = setG16calculator(lig, file_base, label="calculation", WORK_DIR=WORK_DIR)
    elif "uff" in calculator_type.lower():
        if optimization_conf:
            print("UFF calculator not support optimization")
            sys.exit(1)
        else:
            mmCalculator=True

    # set optimizetion parameters
    lig.setOptParams(fmax=thr_fmax, maxiter=args.maxiter)

    if pre_optimization_lig:
        print("Pre-Optimization process.. before confromer generations")
        e = lig.geomOptimization()
        pre_e_file = open("%s/pre_%s%s_energy.txt"%(WORK_DIR, prefix, file_base) , "w")
        print(e, " eV", file=pre_e_file)
        lig.writeRWMol2File("%s/pre_%s%s.sdf"%(WORK_DIR, prefix, file_base), Energy=e)

    if genconformer:
        out_file_path="%s/%sminE_conformer.sdf"%(WORK_DIR, prefix)
        lig = setGenConformers(lig, out_file_path, mmCalculator)
        if lig is None:
            return None
        print("Conformer generation process is done")
    else:
        out_file_path="%s/global_%s%s.sdf"%(WORK_DIR, prefix, file_base)
        # geometry optimizaton for ligand
        if  optimization_lig:
            e = lig.geomOptimization()
            e_file = open("%s/global_%s%s_energy.txt"%(WORK_DIR, prefix, file_base) , "w")
            print(e, " eV", file=e_file)
            lig.writeRWMol2File("%s/global_%s%s.sdf"%(WORK_DIR, prefix, file_base), Energy=e)

    if not verbose:
        _export_final_sdf_and_cleanup(WORK_DIR, file_base, prefix)


if __name__ == "__main__":
    args = parser.parse_args()
    structure_dir = args.structure_dir
    calculator_type = args.calculator_type
    local_model_path = args.local_model_path
    if "nequip" in calculator_type.lower():
        nequip_model_path = local_model_path

    optimization_method = args.optimization_method

    optimization_conf = getBoolStr(args.optimization_conf)
    optimization_lig = getBoolStr(args.optimization_lig)
    pre_optimization_lig = getBoolStr(args.pre_optimization_lig)
    genconformer = getBoolStr(args.genconformer)
    ignore_hydrogen = getBoolStr(args.ignore_hydrogen)
    ETKDG = getBoolStr(args.ETKDG)

    nprocs = args.nprocs
    thr_fmax = args.thr_fmax
    maxiter = args.maxiter

    #get conformer generator parameters
    num_conformers = args.num_conformers
    max_attempts = args.max_attempts
    prune_rms_thresh = args.prune_rms_thresh
    opt_prune_rms_thresh = args.opt_prune_rms_thresh
    opt_prune_diffE_thresh = args.opt_prune_diffE_thresh
    nfold = args.nfold
    npick = args.npick
    nscale = args.nscale

    # optimized-conformer RMSD clustering controls
    cluster_nprocs = args.cluster_nprocs
    cluster_chunk_size = args.cluster_chunk_size
    cluster_linkage = args.cluster_linkage.lower()
    organize_clusters = getBoolStr(args.organize_clusters)
    organize_mode = args.organize_mode.lower()
    summary_csv = args.summary_csv
    verbose = getBoolStr(args.verbose)

    if not verbose:
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.warning')

    file_names = [item for item in os.listdir(structure_dir) if not item.startswith(".")]
    failed_csv = open("failed_files.csv", "w")
    failed_csv.write("FileNames,\n")

    fl_timing = open("timings.csv", "w")
    print("FileName,Time(min.)", file=fl_timing)
    for file_name in file_names:
        file_base = file_name.split(".")[0]
        #  try:
        s_time = time.time()
        result = runConfGen(file_name)
        if result is None:
            continue

        print(file_name, ",", "%.4f"%((time.time()-s_time)/60), file=fl_timing)
        #  except:
        #      print("Error for %s file !!! Skipping...")
        #      failed_csv.write(file_name+",\n")
        #  break
    fl_timing.close()
    failed_csv.close()

