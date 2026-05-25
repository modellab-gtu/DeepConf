#!/usr/bin/env python3
"""
DeepConf validation test matrix — two steps.

Step 1 — Initialization
  Verifies that DeepConf can:
    • add missing hydrogens (no-H input + add_hydrogen=yes)
    • read and write different SDF variants (aromatic, Kekulé)
    • preserve / assign correct bond orders
    • handle charged compounds (+1 / -1)

Step 2 — Processing routes
  Route 1  read + write only (no calc invoked)
  Route 2  geometry optimisation only
  Route 3  conformer generation, no NNP optimisation
  Route 4  conformer generation + NNP optimisation of conformers
  Route 5  pre-optimise input, conformer generation (no conf opt)
  Route 6  pre-optimise input, conformer generation + NNP conf opt
  Route 7  external MD trajectory as conformer source + NNP conf opt
  Route 8  internal ASE Langevin MD + NNP conf opt
  Route 9  pre-optimise input + internal ASE MD + NNP conf opt

Calculator element sets
  ani2x   : H C N O F Cl S
  nequip  : H C N O F Cl S P        (superset of ani2x)
  aimnet2 : 16 elements             (superset of nequip)
  g16     : all elements (Gaussian) — excluded from MD routes (8, 9)

Priority when multiple conformer-source flags are set:
  run_md  >  sample_md  >  genconformer

Examples
--------
Full matrix, aimnet2:
    python scripts/run_deepconf_test_matrix.py --calculator aimnet2

Step 1 only (init checks, aimnet2):
    python scripts/run_deepconf_test_matrix.py --calculator aimnet2 --step 1

Specific routes, benchmarking:
    python scripts/run_deepconf_test_matrix.py --calculator ani2x,aimnet2,nequip \\
        --routes route4,route6 --benchmark

User molecule:
    python scripts/run_deepconf_test_matrix.py --calculator aimnet2 \\
        --input-smiles "your_smiles" --molecule-name mylig
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from rdkit import Chem
from rdkit.Chem import AllChem

# ---------------------------------------------------------------------------
# Calculator element sets
# ---------------------------------------------------------------------------
CALC_ELEMENTS = {
    "ani2x":   {"H", "C", "N", "O", "F", "Cl", "S"},
    "nequip":  {"H", "C", "N", "O", "F", "Cl", "S", "P"},
    "aimnet2": {"H", "C", "N", "O", "F", "Cl", "S", "P",
                "Br", "I", "B", "Si", "Se", "Na", "Ca", "Mg"},
    "g16":     set(),   # Gaussian supports all elements
    "uff":     set(),
}

# ---------------------------------------------------------------------------
# Molecule library
# ---------------------------------------------------------------------------
# no_explicit_h : SDF written without explicit H → DeepConf called with
#                 add_hydrogen=yes (Step 1 init + Step 2 route tests)
# expected_fail_calculators : calculators expected to return non-zero exit
# skip_routes   : route names to skip for this molecule

MOLECULE_LIBRARY = {
    # --- Type A: CHNO only ---------------------------------------------------
    "chno_rigid": {
        "smiles": "CC(=O)Nc1ccc(O)cc1",           # Acetaminophen
        "name":   "Acetaminophen — CHNO, ~2 torsions",
        "charge": 0, "multiplicity": 1,
        "elements": {"H", "C", "N", "O"},
        "compatible_calculators": {"ani2x", "aimnet2", "nequip", "g16", "uff"},
    },
    "chno_rigid_noh": {
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "name":   "Acetaminophen — no explicit H",
        "charge": 0, "multiplicity": 1,
        "elements": {"H", "C", "N", "O"},
        "compatible_calculators": {"ani2x", "aimnet2", "nequip", "g16", "uff"},
        "no_explicit_h": True,
        "skip_routes": {"route8", "route9"},
    },
    "chno_flex": {
        "smiles": "NCCCCCC(=O)O",                  # 6-Aminohexanoic acid
        "name":   "6-Aminohexanoic acid — CHNO, ~5 torsions",
        "charge": 0, "multiplicity": 1,
        "elements": {"H", "C", "N", "O"},
        "compatible_calculators": {"ani2x", "aimnet2", "nequip", "g16", "uff"},
    },
    "chno_flex_noh": {
        "smiles": "NCCCCCC(=O)O",
        "name":   "6-Aminohexanoic acid — no explicit H",
        "charge": 0, "multiplicity": 1,
        "elements": {"H", "C", "N", "O"},
        "compatible_calculators": {"ani2x", "aimnet2", "nequip", "g16", "uff"},
        "no_explicit_h": True,
        "skip_routes": {"route8", "route9"},
    },
    # --- Type B: extends into F/Cl/S/P ---------------------------------------
    "multi_rigid": {
        "smiles": "NS(=O)(=O)c1ccc(Cl)cc1",        # 4-Chlorobenzenesulfonamide
        "name":   "4-Chlorobenzenesulfonamide — CHNOS+Cl, ~1 torsion",
        "charge": 0, "multiplicity": 1,
        "elements": {"H", "C", "N", "O", "S", "Cl"},
        "compatible_calculators": {"ani2x", "aimnet2", "nequip", "g16", "uff"},
    },
    "multi_rigid_noh": {
        "smiles": "NS(=O)(=O)c1ccc(Cl)cc1",
        "name":   "4-Chlorobenzenesulfonamide — no explicit H",
        "charge": 0, "multiplicity": 1,
        "elements": {"H", "C", "N", "O", "S", "Cl"},
        "compatible_calculators": {"ani2x", "aimnet2", "nequip", "g16", "uff"},
        "no_explicit_h": True,
        "skip_routes": {"route8", "route9"},
    },
    "multi_flex": {
        "smiles": "N[C@@H](COP(=O)(O)O)C(=O)O",    # Phosphoserine
        "name":   "Phosphoserine — CHNOP, ~5 torsions",
        "charge": 0, "multiplicity": 1,
        "elements": {"H", "C", "N", "O", "P"},
        "compatible_calculators": {"aimnet2", "nequip", "g16", "uff"},
        "expected_fail_calculators": {"ani2x"},
    },
    "multi_flex_noh": {
        "smiles": "N[C@@H](COP(=O)(O)O)C(=O)O",
        "name":   "Phosphoserine — no explicit H",
        "charge": 0, "multiplicity": 1,
        "elements": {"H", "C", "N", "O", "P"},
        "compatible_calculators": {"aimnet2", "nequip", "g16", "uff"},
        "expected_fail_calculators": {"ani2x"},
        "no_explicit_h": True,
        "skip_routes": {"route8", "route9"},
    },
    # --- Charged -------------------------------------------------------------
    "charged_pos": {
        "smiles": "[NH3+]CC(=O)O",                  # Glycinium  +1
        "name":   "Glycinium (+1) — CHNO",
        "charge": 1, "multiplicity": 1,
        "elements": {"H", "C", "N", "O"},
        "compatible_calculators": {"ani2x", "aimnet2", "nequip", "g16", "uff"},
    },
    "charged_neg": {
        "smiles": "CC(=O)[O-]",                     # Acetate  -1
        "name":   "Acetate (-1) — CHO",
        "charge": -1, "multiplicity": 1,
        "elements": {"H", "C", "O"},
        "compatible_calculators": {"ani2x", "aimnet2", "nequip", "g16", "uff"},
    },
}

# ---------------------------------------------------------------------------
# Route definitions (Step 2)
# ---------------------------------------------------------------------------
# calcs : calculators this route is run against (None = follow --calculator)
# For route1 no output SDF is produced; pass = exit_code == 0.

ROUTE_CASES = {
    "route1": {
        "description": "Read and write only — no calculation",
        "genconformer":        False,
        "optimization_conf":   False,
        "optimization_lig":    False,
        "pre_optimization_lig": False,
        "verbose":             True,
        "organize_clusters":   False,
        "organize_mode":       "move",
        "run_md":              False,
        "sample_md":           False,
        "no_output_expected":  True,    # DeepConf writes nothing in this mode
        "calcs": None,                  # all requested calculators
    },
    "route2": {
        "description": "Geometry optimisation only",
        "genconformer":        False,
        "optimization_conf":   False,
        "optimization_lig":    True,
        "pre_optimization_lig": False,
        "verbose":             True,
        "organize_clusters":   False,
        "organize_mode":       "move",
        "run_md":              False,
        "sample_md":           False,
        "calcs": None,
    },
    "route3": {
        "description": "Conformer generation only — no NNP optimisation",
        "genconformer":        True,
        "optimization_conf":   False,
        "optimization_lig":    False,
        "pre_optimization_lig": False,
        "verbose":             True,
        "organize_clusters":   False,
        "organize_mode":       "move",
        "run_md":              False,
        "sample_md":           False,
        "calcs": None,
    },
    "route4": {
        "description": "Conformer generation + NNP optimisation of conformers",
        "genconformer":        True,
        "optimization_conf":   True,
        "optimization_lig":    False,
        "pre_optimization_lig": False,
        "verbose":             True,
        "organize_clusters":   True,
        "organize_mode":       "move",
        "run_md":              False,
        "sample_md":           False,
        "calcs": None,
    },
    "route5": {
        "description": "Pre-optimise input, conformer generation (no conf opt)",
        "genconformer":        True,
        "optimization_conf":   False,
        "optimization_lig":    False,
        "pre_optimization_lig": True,
        "verbose":             True,
        "organize_clusters":   False,
        "organize_mode":       "move",
        "run_md":              False,
        "sample_md":           False,
        "calcs": None,
    },
    "route6": {
        "description": "Pre-optimise input + conformer generation + NNP conf opt",
        "genconformer":        True,
        "optimization_conf":   True,
        "optimization_lig":    False,
        "pre_optimization_lig": True,
        "verbose":             True,
        "organize_clusters":   True,
        "organize_mode":       "move",
        "run_md":              False,
        "sample_md":           False,
        "calcs": None,
    },
    "route7": {
        "description": "External MD trajectory as conformer source + NNP conf opt",
        "genconformer":        False,
        "optimization_conf":   True,
        "optimization_lig":    False,
        "pre_optimization_lig": False,
        "verbose":             True,
        "organize_clusters":   True,
        "organize_mode":       "move",
        "run_md":              False,
        "sample_md":           True,
        "calcs": None,          # all 4
    },
    "route8": {
        "description": "Internal ASE Langevin MD + NNP conf opt",
        "genconformer":        False,
        "optimization_conf":   True,
        "optimization_lig":    False,
        "pre_optimization_lig": False,
        "verbose":             True,
        "organize_clusters":   True,
        "organize_mode":       "move",
        "run_md":              True,
        "sample_md":           False,
        "calcs": {"ani2x", "aimnet2", "nequip"},   # no g16
    },
    "route9": {
        "description": "Pre-optimise input + internal ASE MD + NNP conf opt",
        "genconformer":        False,
        "optimization_conf":   True,
        "optimization_lig":    False,
        "pre_optimization_lig": True,
        "verbose":             True,
        "organize_clusters":   True,
        "organize_mode":       "move",
        "run_md":              True,
        "sample_md":           False,
        "calcs": {"ani2x", "aimnet2", "nequip"},   # no g16
    },
}

# ---------------------------------------------------------------------------
# Step 1 — initialization tests
# ---------------------------------------------------------------------------
# Each entry maps to a route + molecule.  Runs once with the first
# listed calculator (aimnet2 preferred, fallback to first available).

INIT_TESTS = {
    "init_add_H_rigid": {
        "description": "Add H to rigid CHNO molecule (acetaminophen, no explicit H)",
        "molecule":    "chno_rigid_noh",
        "route":       "route2",
        "capability":  "add_hydrogen",
    },
    "init_add_H_flex": {
        "description": "Add H to flexible CHNO molecule (6-aminohexanoic acid, no H)",
        "molecule":    "chno_flex_noh",
        "route":       "route2",
        "capability":  "add_hydrogen",
    },
    "init_add_H_multi": {
        "description": "Add H to S/Cl molecule (4-ClBenzSulfonamide, no H)",
        "molecule":    "multi_rigid_noh",
        "route":       "route2",
        "capability":  "add_hydrogen",
    },
    "init_bond_orders": {
        "description": "Read aromatic SDF, verify bond orders preserved (acetaminophen)",
        "molecule":    "chno_rigid",
        "route":       "route1",
        "capability":  "bond_orders",
    },
    "init_charged_pos": {
        "description": "Read and optimise positively charged compound (glycinium +1)",
        "molecule":    "charged_pos",
        "route":       "route2",
        "capability":  "charged",
    },
    "init_charged_neg": {
        "description": "Read and optimise negatively charged compound (acetate -1)",
        "molecule":    "charged_neg",
        "route":       "route2",
        "capability":  "charged",
    },
}

# ---------------------------------------------------------------------------
# SDF helpers
# ---------------------------------------------------------------------------

def _str_bool(v):
    return "yes" if v else "no"


def mol_to_sdf(smiles, path, add_h=True, charge=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")
    if add_h:
        mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    mol.SetProp("_Name", Path(path).stem)
    if charge:
        mol.SetProp("FormalCharge", str(charge))
    with Chem.SDWriter(str(path)) as w:
        w.write(mol)


def multi_conf_sdf(smiles, path, n_confs=8):
    """Multi-conformer SDF used as surrogate external MD trajectory."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    ids = [c.GetId() for c in mol.GetConformers()]
    for cid in ids:
        AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=200)
    mol.SetProp("_Name", Path(path).stem)
    with Chem.SDWriter(str(path)) as w:
        for cid in ids:
            w.write(mol, confId=cid)

# ---------------------------------------------------------------------------
# Output location (mirrors DeepConf internals)
# ---------------------------------------------------------------------------

def expected_output_path(case_dir, file_base, route, add_hydrogen):
    wf = route
    if not wf["verbose"]:
        return case_dir / f"{file_base}_output.sdf"
    work = case_dir / file_base
    if not wf["genconformer"] and not wf["run_md"] and not wf["sample_md"]:
        prefix = "addH_" if add_hydrogen else ""
        if wf["optimization_lig"] or wf["optimization_conf"] or wf["pre_optimization_lig"]:
            prefix += "opt_"
        return work / f"global_{prefix}{file_base}.sdf"
    if wf["optimization_conf"]:
        return work / "opt_picked_confs" / f"{file_base}_output.sdf"
    return work / "picked_confs" / f"{file_base}_output.sdf"


def read_sdf_summary(path):
    if not path.exists():
        return {"exists": False, "n_molecules": 0,
                "all_have_energy": False, "same_topology": False}
    mols = [m for m in Chem.SDMolSupplier(str(path), removeHs=False) if m]
    if not mols:
        return {"exists": True, "n_molecules": 0,
                "all_have_energy": False, "same_topology": False}
    n_atoms = mols[0].GetNumAtoms()
    n_bonds = mols[0].GetNumBonds()
    return {
        "exists": True,
        "n_molecules": len(mols),
        "all_have_energy": all(m.HasProp("Energy") for m in mols),
        "same_topology": all(
            m.GetNumAtoms() == n_atoms and m.GetNumBonds() == n_bonds
            for m in mols
        ),
        "atom_count": n_atoms,
    }

# ---------------------------------------------------------------------------
# Command builder — all 45 positional args
# ---------------------------------------------------------------------------

def build_command(args, input_dir, route, mol_info, extra_md_traj=""):
    add_h  = mol_info.get("no_explicit_h", False)
    charge = mol_info.get("charge", 0)
    charge_str = str(charge) if charge != 0 else ""
    mult   = str(mol_info.get("multiplicity", 1))
    run_md    = route.get("run_md", False)
    sample_md = route.get("sample_md", False)
    ext_traj  = extra_md_traj if (sample_md and not run_md) else ""

    return [
        str(args.python),
        str(args.repo_root / "runConfGen.py"),
        str(input_dir),                              # 1  structure_dir
        _str_bool(add_h),                            # 2  ignore_hydrogen
        args.calculator,                             # 3  calculator_type
        args.optimization_method,                    # 4  optimization_method
        _str_bool(route["optimization_conf"]),       # 5  optimization_conf
        _str_bool(route["optimization_lig"]),        # 6  optimization_lig
        _str_bool(route["pre_optimization_lig"]),    # 7  pre_optimization_lig
        _str_bool(route["genconformer"]),            # 8  genconformer
        str(args.nprocs),                            # 9  nprocs
        str(args.thr_fmax),                          # 10 thr_fmax
        str(args.maxiter),                           # 11 maxiter
        _str_bool(args.etkdg),                       # 12 ETKDG
        str(args.num_conformers),                    # 13 num_conformers
        str(args.max_attempts),                      # 14 max_attempts
        str(args.prune_rms_thresh),                  # 15 prune_rms_thresh
        str(args.opt_prune_rms_thresh),              # 16 opt_prune_rms_thresh
        str(args.opt_prune_diffE_thresh),            # 17 opt_prune_diffE_thresh
        str(args.nfold),                             # 18 nfold
        str(args.npick),                             # 19 npick
        str(args.nscale),                            # 20 nscale
        str(args.cluster_nprocs),                    # 21 cluster_nprocs
        str(args.cluster_chunk_size),                # 22 cluster_chunk_size
        args.cluster_linkage,                        # 23 cluster_linkage
        _str_bool(route["organize_clusters"]),       # 24 organize_clusters
        route["organize_mode"],                      # 25 organize_mode
        args.summary_csv,                            # 26 summary_csv
        _str_bool(route["verbose"]),                 # 27 verbose
        args.calculator_model,                       # 28 calculator_model
        charge_str,                                  # 29 calculator_charge
        mult,                                        # 30 calculator_mult
        args.calculator_device,                      # 31 calculator_device
        args.nequip_chemical_symbols,                # 32 nequip_chemical_symbols
        args.g16_mem,                                # 33 g16_mem
        args.g16_level,                              # 34 g16_level
        args.g16_basis,                              # 35 g16_basis
        _str_bool(sample_md),                        # 36 sample_md
        ext_traj,                                    # 37 external_md_traj_file
        _str_bool(run_md),                           # 38 run_md
        str(args.md_temperature),                    # 39 md_temperature
        str(args.md_steps),                          # 40 md_steps
        str(args.md_timestep_fs),                    # 41 md_timestep_fs
        str(args.md_sample_interval),                # 42 md_sample_interval
        str(args.md_friction),                       # 43 md_friction
        str(args.md_box_size),                       # 44 md_box_size
        args.md_traj_file,                           # 45 md_traj_file
    ]

# ---------------------------------------------------------------------------
# Single-case runner
# ---------------------------------------------------------------------------

def run_one(args, label, route, mol_key, mol_info, source_sdf, output_root,
            case_tag=None):
    case_id  = case_tag or f"{label}__{mol_key}__{args.calculator}"
    case_dir = output_root / case_id
    input_dir = case_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    file_base = source_sdf.stem
    shutil.copy2(source_sdf, input_dir / source_sdf.name)

    # External trajectory for route7
    extra_md_traj = ""
    if route.get("sample_md") and not route.get("run_md"):
        traj_sdf = case_dir / "external_traj.sdf"
        multi_conf_sdf(mol_info["smiles"], traj_sdf, n_confs=8)
        extra_md_traj = str(traj_sdf)

    cmd = build_command(args, input_dir, route, mol_info, extra_md_traj)

    fail_calcs = (mol_info.get("expected_fail_calculators") or set())
    expected_fail = args.calculator in fail_calcs

    start = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=str(case_dir), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=args.timeout,
        )
        elapsed = time.time() - start
        timed_out = False
    except subprocess.TimeoutExpired:
        elapsed = args.timeout
        timed_out = True
        proc = type("P", (), {"returncode": "timeout", "stdout": ""})()

    (case_dir / "run.log").write_text(proc.stdout or "")

    add_hydrogen = mol_info.get("no_explicit_h", False)
    no_output    = route.get("no_output_expected", False)
    expected_sdf = expected_output_path(case_dir, file_base, route, add_hydrogen)
    sdf_summary  = read_sdf_summary(expected_sdf) if not no_output else {
        "exists": False, "n_molecules": 0,
        "all_have_energy": False, "same_topology": False,
    }
    work_dir = case_dir / file_base

    if timed_out:
        passed, verdict = False, "TIMEOUT"
    elif expected_fail:
        passed  = proc.returncode != 0
        verdict = "PASS(expected-fail)" if passed else "FAIL(should-have-failed)"
    elif no_output:
        passed  = proc.returncode == 0
        verdict = "PASS" if passed else "FAIL"
    else:
        output_ok  = (sdf_summary["exists"]
                      and sdf_summary["n_molecules"] > 0
                      and sdf_summary["all_have_energy"]
                      and sdf_summary["same_topology"])
        cleanup_ok = route["verbose"] or not work_dir.exists()
        passed  = proc.returncode == 0 and output_ok and cleanup_ok
        verdict = "PASS" if passed else "FAIL"

    return {
        "case_id":       case_id,
        "step":          None,       # filled by caller
        "route":         label,
        "molecule":      mol_key,
        "calculator":    args.calculator,
        "description":   route.get("description", ""),
        "mol_name":      mol_info.get("name", mol_key),
        "passed":        passed,
        "verdict":       verdict,
        "expected_fail": expected_fail,
        "returncode":    proc.returncode,
        "elapsed_s":     round(elapsed, 2),
        "sdf_summary":   sdf_summary,
        "expected_sdf":  str(expected_sdf),
        "log":           str(case_dir / "run.log"),
        "command":       " ".join(cmd),
    }

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_json_report(path, results, args):
    path.write_text(json.dumps({
        "generated":  time.strftime("%Y-%m-%d %H:%M:%S"),
        "calculator": args.calculator,
        "n_passed":   sum(1 for r in results if r["passed"]),
        "n_total":    len(results),
        "results":    results,
    }, indent=2))


def write_markdown_report(path, results):
    lines = [
        "# DeepConf Test Matrix Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Step | Route | Molecule | Calculator | Verdict | RC | "
        "Confs | Energy | Topology | Time(s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        s = r["sdf_summary"]
        lines.append(
            f"| {r['step']} | {r['route']} | {r['molecule']} | {r['calculator']} "
            f"| **{r['verdict']}** | {r['returncode']} "
            f"| {s.get('n_molecules','-')} | {s.get('all_have_energy','-')} "
            f"| {s.get('same_topology','-')} | {r['elapsed_s']} |"
        )
    calcs = sorted({r["calculator"] for r in results})
    if len(calcs) > 1:
        lines += ["", "## Timing by Calculator", "",
                  "| Calculator | Total (s) | Cases | Avg (s) |",
                  "| --- | --- | --- | --- |"]
        for calc in calcs:
            sub   = [r for r in results if r["calculator"] == calc]
            total = sum(r["elapsed_s"] for r in sub)
            avg   = total / len(sub) if sub else 0
            lines.append(f"| {calc} | {total:.1f} | {len(sub)} | {avg:.1f} |")
    path.write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        description="DeepConf validation test matrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    g = p.add_argument_group("Input molecule (overrides built-in library)")
    g.add_argument("--input-smiles",    default=None)
    g.add_argument("--molecule-name",   default="custom")
    g.add_argument("--molecule-charge", type=int, default=0)
    g.add_argument("--input-sdf",       type=Path, default=None)

    g = p.add_argument_group("Test selection")
    g.add_argument("--step", default="1,2",
                   help="Steps to run: 1, 2, or 1,2 (default: 1,2)")
    g.add_argument("--routes", default=None,
                   help="Comma-separated route names to run (default: all)")
    g.add_argument("--molecules", default=None,
                   help="Comma-separated MOLECULE_LIBRARY keys")
    g.add_argument("--calculator", default="aimnet2",
                   help="Comma-separated: ani2x, aimnet2, nequip, g16, uff")

    g = p.add_argument_group("Execution")
    g.add_argument("--repo-root",   type=Path, default=repo_root)
    g.add_argument("--python",      default=sys.executable)
    g.add_argument("--output-root", type=Path, default=None)
    g.add_argument("--timeout",     type=int,  default=3600)
    g.add_argument("--benchmark",   action="store_true")

    g = p.add_argument_group("Calculator")
    g.add_argument("--calculator-model",            default="")
    g.add_argument("--calculator-device",           default="auto")
    g.add_argument("--nequip-chemical-symbols",     default="")
    g.add_argument("--g16-mem",   default="4GB")
    g.add_argument("--g16-level", default="WB97XD")
    g.add_argument("--g16-basis", default="6-311++G(3df,3pd)")

    g = p.add_argument_group("NNP / conformer parameters (production defaults)")
    g.add_argument("--optimization-method",    default="BFGS")
    g.add_argument("--nprocs",                 type=int,   default=1)
    g.add_argument("--thr-fmax",               type=float, default=0.2)
    g.add_argument("--maxiter",                type=int,   default=50000)
    g.add_argument("--no-etkdg", dest="etkdg", action="store_false")
    p.set_defaults(etkdg=True)
    g.add_argument("--num-conformers",         type=int,   default=50)
    g.add_argument("--max-attempts",           type=int,   default=100000)
    g.add_argument("--prune-rms-thresh",       type=float, default=0.05)
    g.add_argument("--opt-prune-rms-thresh",   type=float, default=0.5)
    g.add_argument("--opt-prune-diffE-thresh", type=float, default=0.01)
    g.add_argument("--nfold",                  type=int,   default=2)
    g.add_argument("--npick",                  type=int,   default=0)
    g.add_argument("--nscale",                 type=int,   default=10)
    g.add_argument("--cluster-nprocs",         type=int,   default=8)
    g.add_argument("--cluster-chunk-size",     type=int,   default=4000)
    g.add_argument("--cluster-linkage",        default="complete")
    g.add_argument("--summary-csv",            default="cluster_summary.csv")

    g = p.add_argument_group("MD parameters")
    g.add_argument("--md-temperature",     type=float, default=400.0)
    g.add_argument("--md-steps",           type=int,   default=1000,
                   help="MD steps (default 1000 for testing; 50000 for production)")
    g.add_argument("--md-timestep-fs",     type=float, default=1.0)
    g.add_argument("--md-sample-interval", type=int,   default=100,
                   help="Sample every N steps (default 100 → 10 frames at 1000 steps)")
    g.add_argument("--md-friction",        type=float, default=0.01)
    g.add_argument("--md-box-size",        type=float, default=20.0)
    g.add_argument("--md-traj-file",       default="md_sampled_confs.xyz")

    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args.repo_root = args.repo_root.resolve()

    steps = {int(s.strip()) for s in args.step.split(",") if s.strip().isdigit()}
    calculator_list = [c.strip() for c in args.calculator.split(",") if c.strip()]

    if args.output_root is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_root = args.repo_root / "test_matrix_runs" / stamp
    args.output_root.mkdir(parents=True, exist_ok=True)

    # ---- molecule pool ----
    if args.input_smiles:
        mol_pool = {args.molecule_name: {
            "smiles": args.input_smiles, "name": args.molecule_name,
            "charge": args.molecule_charge, "multiplicity": 1,
            "elements": set(), "compatible_calculators": set(calculator_list),
        }}
    elif args.input_sdf:
        key = args.input_sdf.stem
        mol_pool = {key: {
            "smiles": None, "name": key, "charge": 0, "multiplicity": 1,
            "elements": set(), "compatible_calculators": set(calculator_list),
        }}
    elif args.molecules:
        keys = [k.strip() for k in args.molecules.split(",")]
        mol_pool = {k: MOLECULE_LIBRARY[k] for k in keys if k in MOLECULE_LIBRARY}
    else:
        mol_pool = dict(MOLECULE_LIBRARY)

    # ---- route pool ----
    route_pool = dict(ROUTE_CASES)
    if args.routes:
        keys = {k.strip() for k in args.routes.split(",")}
        route_pool = {k: v for k, v in route_pool.items() if k in keys}

    all_results = []

    for calc in calculator_list:
        args.calculator = calc
        calc_root = args.output_root / calc
        calc_root.mkdir(exist_ok=True)

        # Pre-generate SDF files for mol_pool + any molecules needed by init tests
        sdf_cache = {}
        init_mol_keys = {cfg["molecule"] for cfg in INIT_TESTS.values()}
        sdf_mol_pool = dict(mol_pool)
        for k in init_mol_keys:
            if k not in sdf_mol_pool and k in MOLECULE_LIBRARY:
                sdf_mol_pool[k] = MOLECULE_LIBRARY[k]
        for mol_key, mol_info in sdf_mol_pool.items():
            smiles = mol_info.get("smiles")
            if not smiles:
                if args.input_sdf:
                    sdf_cache[mol_key] = args.input_sdf.resolve()
                continue
            p = calc_root / f"{mol_key}.sdf"
            if not p.exists():
                add_h = not mol_info.get("no_explicit_h", False)
                mol_to_sdf(smiles, p, add_h=add_h,
                           charge=mol_info.get("charge", 0))
            sdf_cache[mol_key] = p

        # ----- Step 1 --------------------------------------------------------
        if 1 in steps:
            print(f"[{calc}] Step 1 — Initialization")
            # Use first available preferred calculator; init always runs on
            # whatever calc is active (tests calc setup + mol reading)
            for init_name, init_cfg in INIT_TESTS.items():
                mol_key = init_cfg["molecule"]
                if mol_key not in MOLECULE_LIBRARY:
                    continue
                mol_info   = MOLECULE_LIBRARY[mol_key]
                route      = ROUTE_CASES[init_cfg["route"]]
                source_sdf = sdf_cache.get(mol_key)
                if source_sdf is None:
                    print(f"  SKIP {init_name}: no SDF")
                    continue
                print(f"  [{init_cfg['capability']}] {init_name} ...", flush=True)
                result = run_one(
                    args, init_cfg["route"], route, mol_key, mol_info,
                    source_sdf, calc_root,
                    case_tag=f"step1__{init_name}__{calc}",
                )
                result["step"] = 1
                result["route"] = f"step1/{init_cfg['route']}"
                result["description"] = init_cfg["description"]
                all_results.append(result)
                status = f"  → {result['verdict']}  ({result['elapsed_s']}s)"
                if not result["passed"]:
                    status += f"  [RC={result['returncode']}]"
                print(status, flush=True)
            print()

        # ----- Step 2 --------------------------------------------------------
        if 2 in steps:
            print(f"[{calc}] Step 2 — Processing routes")
            for route_name, route in route_pool.items():
                # Skip if this route excludes the current calculator
                route_calcs = route.get("calcs")
                if route_calcs is not None and calc not in route_calcs:
                    continue

                for mol_key, mol_info in mol_pool.items():
                    if route_name in (mol_info.get("skip_routes") or set()):
                        continue
                    source_sdf = sdf_cache.get(mol_key)
                    if source_sdf is None:
                        continue

                    label = f"[{calc}] {route_name} / {mol_key}"
                    print(f"  Running {label} ...", flush=True)
                    result = run_one(
                        args, route_name, route, mol_key, mol_info,
                        source_sdf, calc_root,
                    )
                    result["step"] = 2
                    all_results.append(result)
                    status = f"  → {result['verdict']}  ({result['elapsed_s']}s)"
                    if not result["passed"]:
                        status += f"  [RC={result['returncode']}]"
                    print(status, flush=True)
            print()

    # ---- reports ----
    json_path = args.output_root / "test_matrix_report.json"
    md_path   = args.output_root / "test_matrix_report.md"
    write_json_report(json_path, all_results, args)
    write_markdown_report(md_path, all_results)

    n_pass  = sum(1 for r in all_results if r["passed"])
    n_total = len(all_results)
    print(f"Results: {n_pass}/{n_total} passed")
    print(f"JSON    : {json_path}")
    print(f"Markdown: {md_path}")

    if args.benchmark and all_results:
        print("\nBenchmark (slowest first):")
        print(f"  {'Case':<65} {'s':>6}")
        print(f"  {'-'*65} {'-'*6}")
        for r in sorted(all_results, key=lambda x: -x["elapsed_s"]):
            label = f"{r['calculator']}/{r['route']}/{r['molecule']}"
            print(f"  {label:<65} {r['elapsed_s']:>6.1f}")

    failed = [r for r in all_results if not r["passed"]]
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for r in failed:
            print(f"  {r['case_id']}  RC={r['returncode']}  log: {r['log']}")

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
