#!/usr/bin/env python3
"""
Comprehensive DeepConf test matrix.

Tests DeepConf across molecule diversity (small/large, neutral/charged,
varied element sets), all workflow modes, MD sampling, edge cases, and
multiple calculators — tiered from fast/easy to thorough/hard.

Examples
--------
Tier-1 sanity check on aimnet2 (fast, ~1 min):
    python scripts/run_deepconf_test_matrix.py --tier 1 --calculator aimnet2

Full tier-1+2 matrix:
    python scripts/run_deepconf_test_matrix.py --tier 1,2 --calculator aimnet2

User-supplied SMILES with benchmarking across calculators:
    python scripts/run_deepconf_test_matrix.py \\
        --input-smiles "c1cccnc1" --molecule-name pyridine \\
        --calculator ani2x,aimnet2,nequip --benchmark

All tiers + all built-in molecules:
    python scripts/run_deepconf_test_matrix.py --full --calculator aimnet2

User-supplied SDF:
    python scripts/run_deepconf_test_matrix.py \\
        --input-sdf /path/to/ligand.sdf --tier 1,2 --calculator aimnet2
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
# Built-in molecule library
# ---------------------------------------------------------------------------
# tier: 1=easy (fast, broadest calc support), 2=medium, 3=edge cases
# compatible_calculators: set of calculators known to support this molecule
# expected_fail_calculators: set of calculators expected to fail (wrong elements)
# no_explicit_h: write SDF without explicit H atoms (to test add_hydrogen flow)

MOLECULE_LIBRARY = {
    # --- Tier 1: tiny/small neutral HCNO ---------------------------------
    "ethanol": {
        "smiles": "CCO",
        "name": "Ethanol",
        "tier": 1,
        "charge": 0,
        "multiplicity": 1,
        "elements": {"H", "C", "O"},
        "compatible_calculators": {"ani2x", "aimnet2", "nequip", "uff"},
        "notes": "Tiny 3-heavy-atom neutral baseline",
    },
    "aspirin": {
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "name": "Aspirin",
        "tier": 1,
        "charge": 0,
        "multiplicity": 1,
        "elements": {"H", "C", "O"},
        "compatible_calculators": {"ani2x", "aimnet2", "nequip", "uff"},
        "notes": "Small aromatic HCNO",
    },
    "acetaminophen": {
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "name": "Acetaminophen",
        "tier": 1,
        "charge": 0,
        "multiplicity": 1,
        "elements": {"H", "C", "N", "O"},
        "compatible_calculators": {"ani2x", "aimnet2", "nequip", "uff"},
        "notes": "HCNO amide+phenol — all ANI2x-safe elements",
    },
    # --- Tier 2: varied elements, more flexibility -----------------------
    "methionine": {
        "smiles": "CSCC[C@@H](N)C(=O)O",
        "name": "Methionine",
        "tier": 2,
        "charge": 0,
        "multiplicity": 1,
        "elements": {"H", "C", "N", "O", "S"},
        "compatible_calculators": {"ani2x", "aimnet2", "nequip", "uff"},
        "notes": "Sulfur-containing amino acid (HCNOS)",
    },
    "4_fluoroaniline": {
        "smiles": "Nc1ccc(F)cc1",
        "name": "4-Fluoroaniline",
        "tier": 2,
        "charge": 0,
        "multiplicity": 1,
        "elements": {"H", "C", "N", "F"},
        "compatible_calculators": {"ani2x", "aimnet2", "nequip", "uff"},
        "notes": "Fluorine substituent (F supported by ANI2x)",
    },
    "4_chloroaniline": {
        "smiles": "Nc1ccc(Cl)cc1",
        "name": "4-Chloroaniline",
        "tier": 2,
        "charge": 0,
        "multiplicity": 1,
        "elements": {"H", "C", "N", "Cl"},
        "compatible_calculators": {"ani2x", "aimnet2", "nequip", "uff"},
        "notes": "Chlorine substituent (Cl supported by ANI2x)",
    },
    "ibuprofen": {
        "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "name": "Ibuprofen",
        "tier": 2,
        "charge": 0,
        "multiplicity": 1,
        "elements": {"H", "C", "O"},
        "compatible_calculators": {"ani2x", "aimnet2", "nequip", "uff"},
        "notes": "Medium flexible NSAID, good conformational diversity",
    },
    # --- Tier 3: edge cases (charged, unusual elements, missing H) -------
    "methylammonium": {
        "smiles": "[NH3+]C",
        "name": "Methylammonium (+1)",
        "tier": 3,
        "charge": 1,
        "multiplicity": 1,
        "elements": {"H", "C", "N"},
        "compatible_calculators": {"ani2x", "aimnet2", "nequip", "uff"},
        "notes": "Positively charged (+1) — tests charge forwarding",
    },
    "acetate": {
        "smiles": "CC(=O)[O-]",
        "name": "Acetate (-1)",
        "tier": 3,
        "charge": -1,
        "multiplicity": 1,
        "elements": {"H", "C", "O"},
        "compatible_calculators": {"ani2x", "aimnet2", "nequip", "uff"},
        "notes": "Negatively charged (-1) — tests charge forwarding",
    },
    "bromobenzene": {
        "smiles": "Brc1ccccc1",
        "name": "Bromobenzene",
        "tier": 3,
        "charge": 0,
        "multiplicity": 1,
        "elements": {"H", "C", "Br"},
        "compatible_calculators": {"aimnet2", "nequip", "uff"},
        "expected_fail_calculators": {"ani2x"},
        "notes": "Br not in ANI2x set — expected-failure probe",
    },
    "phosphoserine": {
        "smiles": "N[C@@H](COP(=O)(O)O)C(=O)O",
        "name": "Phosphoserine",
        "tier": 3,
        "charge": 0,
        "multiplicity": 1,
        "elements": {"H", "C", "N", "O", "P"},
        "compatible_calculators": {"aimnet2", "nequip", "uff"},
        "expected_fail_calculators": {"ani2x"},
        "notes": "P not in ANI2x set — expected-failure probe",
    },
    "benzene_noh": {
        "smiles": "c1ccccc1",
        "name": "Benzene (no explicit H)",
        "tier": 3,
        "charge": 0,
        "multiplicity": 1,
        "elements": {"H", "C"},
        "compatible_calculators": {"ani2x", "aimnet2", "nequip", "uff"},
        "notes": "SDF without explicit H — tests add_hydrogen=yes path",
        "no_explicit_h": True,
    },
}

# ---------------------------------------------------------------------------
# Workflow case definitions
# ---------------------------------------------------------------------------
# Each entry maps to a specific combination of DeepConf flags.
# molecule_override: always use this molecule key regardless of --molecules.
# expected_fail_calculators: calculators expected to return non-zero exit code.

WORKFLOW_CASES = {
    # Tier 1 — fast sanity (~seconds each on small molecule) --------------
    "t1_geom_only": {
        "tier": 1,
        "description": "Geometry optimization only, no conformer generation",
        "genconformer": False,
        "optimization_conf": False,
        "optimization_lig": True,
        "pre_optimization_lig": False,
        "verbose": True,
        "organize_clusters": False,
        "organize_mode": "move",
        "run_md": False,
        "sample_md": False,
        "add_hydrogen": False,
    },
    "t1_confs_no_opt": {
        "tier": 1,
        "description": "RDKit conformer generation only, no NNP optimization",
        "genconformer": True,
        "optimization_conf": False,
        "optimization_lig": False,
        "pre_optimization_lig": False,
        "verbose": True,
        "organize_clusters": False,
        "organize_mode": "move",
        "run_md": False,
        "sample_md": False,
        "add_hydrogen": False,
    },
    # Tier 2 — standard workflows (minutes on small molecule) -------------
    "t2_confs_with_opt": {
        "tier": 2,
        "description": "Conformer generation + NNP optimization, verbose=yes",
        "genconformer": True,
        "optimization_conf": True,
        "optimization_lig": False,
        "pre_optimization_lig": False,
        "verbose": True,
        "organize_clusters": True,
        "organize_mode": "move",
        "run_md": False,
        "sample_md": False,
        "add_hydrogen": False,
    },
    "t2_preopt_confs_opt": {
        "tier": 2,
        "description": "Pre-optimize ligand, then generate and optimize conformers",
        "genconformer": True,
        "optimization_conf": True,
        "optimization_lig": False,
        "pre_optimization_lig": True,
        "verbose": True,
        "organize_clusters": True,
        "organize_mode": "move",
        "run_md": False,
        "sample_md": False,
        "add_hydrogen": False,
    },
    "t2_confs_opt_verbose_no": {
        "tier": 2,
        "description": "Confs + opt, verbose=no — work dir should be cleaned up",
        "genconformer": True,
        "optimization_conf": True,
        "optimization_lig": False,
        "pre_optimization_lig": False,
        "verbose": False,
        "organize_clusters": True,
        "organize_mode": "move",
        "run_md": False,
        "sample_md": False,
        "add_hydrogen": False,
    },
    "t2_confs_opt_org_copy": {
        "tier": 2,
        "description": "Confs + opt, cluster organization in copy mode",
        "genconformer": True,
        "optimization_conf": True,
        "optimization_lig": False,
        "pre_optimization_lig": False,
        "verbose": True,
        "organize_clusters": True,
        "organize_mode": "copy",
        "run_md": False,
        "sample_md": False,
        "add_hydrogen": False,
    },
    "t2_geom_only_verbose_no": {
        "tier": 2,
        "description": "Geometry optimization, verbose=no — tests cleanup path",
        "genconformer": False,
        "optimization_conf": False,
        "optimization_lig": True,
        "pre_optimization_lig": False,
        "verbose": False,
        "organize_clusters": False,
        "organize_mode": "move",
        "run_md": False,
        "sample_md": False,
        "add_hydrogen": False,
    },
    # Tier 3 — advanced features and edge cases ---------------------------
    "t3_run_md": {
        "tier": 3,
        "description": "ASE Langevin MD (100 steps, interval 10 → ~10 frames as conformers)",
        "genconformer": False,
        "optimization_conf": True,
        "optimization_lig": False,
        "pre_optimization_lig": False,
        "verbose": True,
        "organize_clusters": True,
        "organize_mode": "move",
        "run_md": True,
        "md_steps": 100,
        "md_sample_interval": 10,
        "md_temperature": 400,
        "sample_md": False,
        "add_hydrogen": False,
    },
    "t3_sample_md_sdf": {
        "tier": 3,
        "description": "External SDF trajectory (multi-conformer) as conformer source",
        "genconformer": False,
        "optimization_conf": True,
        "optimization_lig": False,
        "pre_optimization_lig": False,
        "verbose": True,
        "organize_clusters": True,
        "organize_mode": "move",
        "run_md": False,
        "sample_md": True,
        "add_hydrogen": False,
    },
    "t3_add_hydrogen": {
        "tier": 3,
        "description": "Molecule without explicit H in SDF — add_hydrogen=yes path",
        "genconformer": True,
        "optimization_conf": True,
        "optimization_lig": False,
        "pre_optimization_lig": False,
        "verbose": True,
        "organize_clusters": True,
        "organize_mode": "move",
        "run_md": False,
        "sample_md": False,
        "add_hydrogen": True,
        "molecule_override": "benzene_noh",
    },
    "t3_charged_pos": {
        "tier": 3,
        "description": "Positively charged molecule (+1) — charge forwarded to calculator",
        "genconformer": True,
        "optimization_conf": True,
        "optimization_lig": False,
        "pre_optimization_lig": False,
        "verbose": True,
        "organize_clusters": True,
        "organize_mode": "move",
        "run_md": False,
        "sample_md": False,
        "add_hydrogen": False,
        "molecule_override": "methylammonium",
    },
    "t3_charged_neg": {
        "tier": 3,
        "description": "Negatively charged molecule (-1) — charge forwarded to calculator",
        "genconformer": True,
        "optimization_conf": True,
        "optimization_lig": False,
        "pre_optimization_lig": False,
        "verbose": True,
        "organize_clusters": True,
        "organize_mode": "move",
        "run_md": False,
        "sample_md": False,
        "add_hydrogen": False,
        "molecule_override": "acetate",
    },
    "t3_incompatible_P": {
        "tier": 3,
        "description": "Phosphorus-containing molecule — expected failure with ANI2x",
        "genconformer": True,
        "optimization_conf": True,
        "optimization_lig": False,
        "pre_optimization_lig": False,
        "verbose": True,
        "organize_clusters": False,
        "organize_mode": "move",
        "run_md": False,
        "sample_md": False,
        "add_hydrogen": False,
        "molecule_override": "phosphoserine",
        "expected_fail_calculators": {"ani2x"},
    },
    "t3_incompatible_Br": {
        "tier": 3,
        "description": "Bromine-containing molecule — expected failure with ANI2x",
        "genconformer": True,
        "optimization_conf": True,
        "optimization_lig": False,
        "pre_optimization_lig": False,
        "verbose": True,
        "organize_clusters": False,
        "organize_mode": "move",
        "run_md": False,
        "sample_md": False,
        "add_hydrogen": False,
        "molecule_override": "bromobenzene",
        "expected_fail_calculators": {"ani2x"},
    },
}

# ---------------------------------------------------------------------------
# SDF / molecule helpers
# ---------------------------------------------------------------------------

def _str_bool(value):
    return "yes" if value else "no"


def mol_to_sdf(smiles, path, add_h=True, charge=None):
    """Generate a single-conformer SDF from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    if add_h:
        mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    mol.SetProp("_Name", Path(path).stem)
    if charge is not None and charge != 0:
        mol.SetProp("FormalCharge", str(charge))
    with Chem.SDWriter(str(path)) as w:
        w.write(mol)


def multi_conf_sdf(smiles, path, n_confs=8, add_h=True):
    """Generate an SDF file with multiple conformers (external trajectory stand-in)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    if add_h:
        mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    conf_ids = [c.GetId() for c in mol.GetConformers()]
    for cid in conf_ids:
        AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=100)
    mol.SetProp("_Name", Path(path).stem)
    with Chem.SDWriter(str(path)) as w:
        for cid in conf_ids:
            w.write(mol, confId=cid)


# ---------------------------------------------------------------------------
# Output location logic
# ---------------------------------------------------------------------------

def expected_output_path(case_dir, file_base, wf):
    """Predict the SDF path that DeepConf should write."""
    if not wf["verbose"]:
        return case_dir / f"{file_base}_output.sdf"

    work_dir = case_dir / file_base
    add_hydrogen = wf.get("add_hydrogen", False)

    if not wf["genconformer"] and not wf.get("run_md") and not wf.get("sample_md"):
        prefix = ""
        if add_hydrogen:
            prefix += "addH_"
        if wf["optimization_lig"] or wf["optimization_conf"] or wf["pre_optimization_lig"]:
            prefix += "opt_"
        return work_dir / f"global_{prefix}{file_base}.sdf"

    if wf["optimization_conf"]:
        return work_dir / "opt_picked_confs" / f"{file_base}_output.sdf"

    return work_dir / "picked_confs" / f"{file_base}_output.sdf"


def read_sdf_summary(path):
    if not path.exists():
        return {"exists": False, "n_molecules": 0, "all_have_energy": False,
                "same_topology": False}
    mols = [m for m in Chem.SDMolSupplier(str(path), removeHs=False) if m is not None]
    if not mols:
        return {"exists": True, "n_molecules": 0, "all_have_energy": False,
                "same_topology": False}
    n_atoms = mols[0].GetNumAtoms()
    n_bonds = mols[0].GetNumBonds()
    return {
        "exists": True,
        "n_molecules": len(mols),
        "all_have_energy": all(m.HasProp("Energy") for m in mols),
        "same_topology": all(
            m.GetNumAtoms() == n_atoms and m.GetNumBonds() == n_bonds for m in mols
        ),
        "atom_count": n_atoms,
        "bond_count": n_bonds,
    }


# ---------------------------------------------------------------------------
# Command builder — all 45 positional args
# ---------------------------------------------------------------------------

def build_command(args, input_dir, wf, mol_info, extra_md_traj=""):
    charge = str(mol_info.get("charge", "") or "")
    if charge == "0":
        charge = ""
    md_steps = wf.get("md_steps", args.md_steps)
    md_sample_interval = wf.get("md_sample_interval", args.md_sample_interval)
    md_temperature = wf.get("md_temperature", args.md_temperature)
    sample_md = wf.get("sample_md", False)
    run_md = wf.get("run_md", False)
    external_traj = extra_md_traj if sample_md else ""

    return [
        str(args.python),
        str(args.repo_root / "runConfGen.py"),
        str(input_dir),                              # 1  structure_dir
        _str_bool(wf.get("add_hydrogen", False)),    # 2  ignore_hydrogen
        args.calculator,                             # 3  calculator_type
        args.optimization_method,                    # 4  optimization_method
        _str_bool(wf["optimization_conf"]),          # 5  optimization_conf
        _str_bool(wf["optimization_lig"]),           # 6  optimization_lig
        _str_bool(wf["pre_optimization_lig"]),       # 7  pre_optimization_lig
        _str_bool(wf["genconformer"]),               # 8  genconformer
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
        _str_bool(wf["organize_clusters"]),          # 24 organize_clusters
        wf["organize_mode"],                         # 25 organize_mode
        args.summary_csv,                            # 26 summary_csv
        _str_bool(wf["verbose"]),                    # 27 verbose
        args.calculator_model,                       # 28 calculator_model
        charge,                                      # 29 calculator_charge
        str(mol_info.get("multiplicity", 1)),        # 30 calculator_mult
        args.calculator_device,                      # 31 calculator_device
        args.nequip_chemical_symbols,                # 32 nequip_chemical_symbols
        args.g16_mem,                                # 33 g16_mem
        args.g16_level,                              # 34 g16_level
        args.g16_basis,                              # 35 g16_basis
        _str_bool(sample_md),                        # 36 sample_md
        external_traj,                               # 37 external_md_traj_file
        _str_bool(run_md),                           # 38 run_md
        str(md_temperature),                         # 39 md_temperature
        str(md_steps),                               # 40 md_steps
        str(args.md_timestep_fs),                    # 41 md_timestep_fs
        str(md_sample_interval),                     # 42 md_sample_interval
        str(args.md_friction),                       # 43 md_friction
        str(args.md_box_size),                       # 44 md_box_size
        args.md_traj_file,                           # 45 md_traj_file
    ]


# ---------------------------------------------------------------------------
# Test case runner
# ---------------------------------------------------------------------------

def run_one(args, wf_name, wf, mol_key, mol_info, source_sdf, output_root):
    case_id = f"{wf_name}__{mol_key}__{args.calculator}"
    case_dir = output_root / case_id
    input_dir = case_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Input SDF — molecules that need no explicit H get a stripped SDF
    if mol_info.get("no_explicit_h"):
        mol_sdf = input_dir / f"{mol_key}.sdf"
        mol_to_sdf(mol_info["smiles"], mol_sdf, add_h=False)
        file_base = mol_key
    else:
        mol_sdf = input_dir / source_sdf.name
        shutil.copy2(source_sdf, mol_sdf)
        file_base = source_sdf.stem

    # External trajectory SDF for sample_md cases — match the input topology's H count
    extra_md_traj = ""
    if wf.get("sample_md"):
        traj_sdf = case_dir / "external_traj.sdf"
        traj_add_h = not mol_info.get("no_explicit_h", False)
        multi_conf_sdf(mol_info["smiles"], traj_sdf, n_confs=8, add_h=traj_add_h)
        extra_md_traj = str(traj_sdf)

    cmd = build_command(args, input_dir, wf, mol_info, extra_md_traj)

    # Is this case expected to fail?
    wf_fail_calcs = wf.get("expected_fail_calculators", set())
    mol_fail_calcs = mol_info.get("expected_fail_calculators", set())
    expected_fail = args.calculator in (wf_fail_calcs | mol_fail_calcs)

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(case_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
        )
        elapsed = time.time() - start
        timed_out = False
    except subprocess.TimeoutExpired:
        elapsed = args.timeout
        timed_out = True
        proc = type("P", (), {"returncode": "timeout", "stdout": ""})()

    log_path = case_dir / "run.log"
    log_path.write_text(proc.stdout or "")

    expected_sdf = expected_output_path(case_dir, file_base, wf)
    sdf_summary = read_sdf_summary(expected_sdf)
    work_dir = case_dir / file_base

    if timed_out:
        passed = False
        verdict = "TIMEOUT"
    elif expected_fail:
        # For expected-failure cases pass = run returned non-zero
        passed = proc.returncode != 0
        verdict = "PASS(expected fail)" if passed else "FAIL(should have failed)"
    else:
        output_ok = (
            sdf_summary["exists"]
            and sdf_summary["n_molecules"] > 0
            and sdf_summary["all_have_energy"]
            and sdf_summary["same_topology"]
        )
        verbose_cleanup_ok = wf["verbose"] or not work_dir.exists()
        passed = proc.returncode == 0 and output_ok and verbose_cleanup_ok
        verdict = "PASS" if passed else "FAIL"

    return {
        "case_id": case_id,
        "workflow": wf_name,
        "molecule": mol_key,
        "calculator": args.calculator,
        "tier": wf["tier"],
        "passed": passed,
        "verdict": verdict,
        "expected_fail": expected_fail,
        "returncode": proc.returncode,
        "elapsed_s": round(elapsed, 2),
        "sdf_summary": sdf_summary,
        "expected_sdf": str(expected_sdf),
        "log": str(log_path),
        "command": " ".join(cmd),
        "description": wf.get("description", ""),
        "mol_notes": mol_info.get("notes", ""),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_json_report(path, results, args):
    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "calculator": args.calculator,
        "tiers": sorted(args.tiers),
        "python": str(args.python),
        "n_passed": sum(1 for r in results if r["passed"]),
        "n_total": len(results),
        "results": results,
    }
    path.write_text(json.dumps(report, indent=2))


def write_markdown_report(path, results):
    lines = [
        "# DeepConf Test Matrix Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Tier | Workflow | Molecule | Calculator | Verdict | RC | "
        "Confs | Energy | Topology | Time(s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        s = r["sdf_summary"]
        lines.append(
            f"| {r['tier']} | {r['workflow']} | {r['molecule']} | {r['calculator']} "
            f"| **{r['verdict']}** | {r['returncode']} "
            f"| {s.get('n_molecules', '-')} | {s.get('all_have_energy', '-')} "
            f"| {s.get('same_topology', '-')} | {r['elapsed_s']} |"
        )

    # Benchmark summary if multiple calculators/molecules
    calcs = sorted({r["calculator"] for r in results})
    if len(calcs) > 1:
        lines += ["", "## Timing by Calculator (sum s)", ""]
        lines.append("| Calculator | Total (s) | Cases | Avg (s) |")
        lines.append("| --- | --- | --- | --- |")
        for calc in calcs:
            sub = [r for r in results if r["calculator"] == calc]
            total = sum(r["elapsed_s"] for r in sub)
            avg = total / len(sub) if sub else 0
            lines.append(f"| {calc} | {total:.1f} | {len(sub)} | {avg:.1f} |")

    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        description="Comprehensive DeepConf test matrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input molecule
    g_mol = p.add_argument_group("Input molecule")
    g_mol.add_argument("--input-sdf", type=Path, default=None,
                       help="User-supplied SDF (overrides built-in library for non-override cases)")
    g_mol.add_argument("--input-smiles", default=None,
                       help="Generate SDF on-the-fly from this SMILES")
    g_mol.add_argument("--molecule-name", default="custom",
                       help="Name to use for --input-smiles molecule")
    g_mol.add_argument("--molecule-charge", type=int, default=0,
                       help="Formal charge for --input-smiles molecule")
    g_mol.add_argument("--molecules", default=None,
                       help="Comma-separated keys from built-in library to include "
                            "(default: ethanol for tier1, tier1+2 molecules for tier2+, "
                            "all for --full)")

    # Test selection
    g_sel = p.add_argument_group("Test selection")
    g_sel.add_argument("--tier", dest="tier_str", default="1",
                       help="Comma-separated tiers to run: 1, 2, 3, or 'all' (default: 1)")
    g_sel.add_argument("--full", action="store_true",
                       help="Run all tiers on all built-in molecules (overrides --tier/--molecules)")
    g_sel.add_argument("--workflow-filter", action="append", default=[],
                       help="Substring filter on workflow name (can repeat)")

    # Calculator
    g_calc = p.add_argument_group("Calculator")
    g_calc.add_argument("--calculator", default="aimnet2",
                        help="Comma-separated calculators: ani2x, aimnet2, nequip, uff, g16")
    g_calc.add_argument("--calculator-model", default="",
                        help="Path to NequIP model or AIMNet2 variant")
    g_calc.add_argument("--calculator-device", default="auto")
    g_calc.add_argument("--nequip-chemical-symbols", default="")
    g_calc.add_argument("--g16-mem", default="4GB")
    g_calc.add_argument("--g16-level", default="WB97XD")
    g_calc.add_argument("--g16-basis", default="6-311++G(3df,3pd)")

    # Execution
    g_exec = p.add_argument_group("Execution")
    g_exec.add_argument("--repo-root", type=Path, default=repo_root)
    g_exec.add_argument("--python", default=sys.executable)
    g_exec.add_argument("--output-root", type=Path, default=None)
    g_exec.add_argument("--timeout", type=int, default=900,
                        help="Per-case timeout in seconds (default: 900)")
    g_exec.add_argument("--benchmark", action="store_true",
                        help="Print per-case timing table at the end")

    # NNP / conformer parameters (kept minimal for speed; override as needed)
    g_nnp = p.add_argument_group("NNP / conformer parameters")
    g_nnp.add_argument("--optimization-method", default="BFGS")
    g_nnp.add_argument("--nprocs", type=int, default=1)
    g_nnp.add_argument("--thr-fmax", type=float, default=10.0,
                        help="Force threshold for optimization (loose default for speed)")
    g_nnp.add_argument("--maxiter", type=int, default=5,
                        help="Max optimization iterations (small default for speed)")
    g_nnp.add_argument("--no-etkdg", dest="etkdg", action="store_false")
    p.set_defaults(etkdg=True)
    g_nnp.add_argument("--num-conformers", type=int, default=4,
                        help="Number of conformers to generate (default: 4 for speed)")
    g_nnp.add_argument("--max-attempts", type=int, default=50)
    g_nnp.add_argument("--prune-rms-thresh", type=float, default=0.5)
    g_nnp.add_argument("--opt-prune-rms-thresh", type=float, default=0.5)
    g_nnp.add_argument("--opt-prune-diffE-thresh", type=float, default=0.001)
    g_nnp.add_argument("--nfold", type=int, default=1)
    g_nnp.add_argument("--npick", type=int, default=0)
    g_nnp.add_argument("--nscale", type=int, default=1)
    g_nnp.add_argument("--cluster-nprocs", type=int, default=1)
    g_nnp.add_argument("--cluster-chunk-size", type=int, default=50)
    g_nnp.add_argument("--cluster-linkage", default="complete")
    g_nnp.add_argument("--summary-csv", default="cluster_summary.csv")

    # MD parameters
    g_md = p.add_argument_group("MD parameters (for t3_run_md cases)")
    g_md.add_argument("--md-temperature", type=float, default=400.0)
    g_md.add_argument("--md-steps", type=int, default=100,
                       help="MD steps (default: 100 for fast testing)")
    g_md.add_argument("--md-timestep-fs", type=float, default=1.0)
    g_md.add_argument("--md-sample-interval", type=int, default=10,
                       help="Sample every N MD steps (default: 10)")
    g_md.add_argument("--md-friction", type=float, default=0.01)
    g_md.add_argument("--md-box-size", type=float, default=20.0)
    g_md.add_argument("--md-traj-file", default="md_sampled_confs.xyz")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args.repo_root = args.repo_root.resolve()

    # Parse tiers
    if args.full:
        args.tiers = {1, 2, 3}
    elif args.tier_str.lower() == "all":
        args.tiers = {1, 2, 3}
    else:
        try:
            args.tiers = {int(t.strip()) for t in args.tier_str.split(",")}
        except ValueError:
            print(f"ERROR: --tier must be comma-separated integers or 'all', got: {args.tier_str}")
            return 1

    # Parse calculators (support comma-separated list or single value)
    calculator_list = [c.strip() for c in args.calculator.split(",") if c.strip()]

    # Output root
    if args.output_root is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_root = args.repo_root / "test_matrix_runs" / stamp
    args.output_root.mkdir(parents=True, exist_ok=True)

    # Determine primary molecule source
    if args.input_smiles:
        mol_key = args.molecule_name
        mol_info = {
            "smiles": args.input_smiles,
            "name": args.molecule_name,
            "tier": 1,
            "charge": args.molecule_charge,
            "multiplicity": 1,
            "elements": set(),
            "compatible_calculators": set(calculator_list),
            "notes": f"User-supplied SMILES: {args.input_smiles}",
        }
        source_sdf = args.output_root / f"{mol_key}.sdf"
        mol_to_sdf(args.input_smiles, source_sdf,
                   charge=args.molecule_charge)
        primary_molecules = {mol_key: mol_info}
    elif args.input_sdf:
        mol_key = args.input_sdf.stem
        mol_info = {
            "smiles": None,
            "name": mol_key,
            "tier": 1,
            "charge": 0,
            "multiplicity": 1,
            "elements": set(),
            "compatible_calculators": set(calculator_list),
            "notes": f"User-supplied SDF: {args.input_sdf}",
        }
        source_sdf = args.input_sdf.resolve()
        primary_molecules = {mol_key: mol_info}
    else:
        # Pick built-in molecules based on --molecules flag or tier defaults
        if args.molecules:
            keys = [k.strip() for k in args.molecules.split(",")]
            primary_molecules = {k: MOLECULE_LIBRARY[k] for k in keys
                                  if k in MOLECULE_LIBRARY}
        elif args.full:
            primary_molecules = {k: v for k, v in MOLECULE_LIBRARY.items()
                                  if v["tier"] <= max(args.tiers)}
        else:
            # Default: include molecules at or below the max requested tier
            max_tier = max(args.tiers)
            if max_tier == 1:
                default_keys = ["ethanol", "aspirin"]
            elif max_tier == 2:
                default_keys = ["ethanol", "aspirin", "acetaminophen",
                                "methionine", "ibuprofen"]
            else:
                default_keys = list(MOLECULE_LIBRARY.keys())
            primary_molecules = {k: MOLECULE_LIBRARY[k] for k in default_keys}
        source_sdf = None  # will be generated per molecule

    # Filter workflow cases
    selected_workflows = {
        name: wf for name, wf in WORKFLOW_CASES.items()
        if wf["tier"] in args.tiers
    }
    if args.workflow_filter:
        selected_workflows = {
            name: wf for name, wf in selected_workflows.items()
            if any(f in name for f in args.workflow_filter)
        }
    if not selected_workflows:
        print("ERROR: No workflow cases match the selected tiers/filters.")
        return 1

    # Build flat task list
    tasks = []
    for wf_name, wf in selected_workflows.items():
        override = wf.get("molecule_override")
        if override:
            # Fixed molecule regardless of --molecules
            mol_info_override = MOLECULE_LIBRARY.get(override)
            if mol_info_override is None:
                print(f"WARNING: molecule_override '{override}' not in library, skipping {wf_name}")
                continue
            tasks.append((wf_name, wf, override, mol_info_override))
        else:
            for mk, mi in primary_molecules.items():
                tasks.append((wf_name, wf, mk, mi))

    # Deduplicate tasks for the same (workflow, molecule)
    seen = set()
    unique_tasks = []
    for wf_name, wf, mk, mi in tasks:
        key = (wf_name, mk)
        if key not in seen:
            seen.add(key)
            unique_tasks.append((wf_name, wf, mk, mi))
    tasks = unique_tasks

    print(f"Test matrix: {len(tasks)} cases × {len(calculator_list)} calculator(s) = "
          f"{len(tasks) * len(calculator_list)} total runs")
    print(f"Output root: {args.output_root}")
    print()

    all_results = []

    for calc in calculator_list:
        args.calculator = calc
        calc_root = args.output_root / calc
        calc_root.mkdir(exist_ok=True)

        for wf_name, wf, mol_key, mol_info in tasks:
            # Prepare source SDF for non-override molecules
            if "molecule_override" not in wf and source_sdf is None:
                sdf_path = calc_root / f"{mol_key}.sdf"
                if not sdf_path.exists():
                    if mol_info.get("smiles"):
                        mol_to_sdf(mol_info["smiles"], sdf_path,
                                   charge=mol_info.get("charge", 0))
                    else:
                        print(f"  SKIP {wf_name}/{mol_key}: no SDF source")
                        continue
                effective_sdf = sdf_path
            elif "molecule_override" in wf:
                sdf_path = calc_root / f"{mol_key}.sdf"
                if not sdf_path.exists():
                    mol_to_sdf(mol_info["smiles"], sdf_path,
                               charge=mol_info.get("charge", 0))
                effective_sdf = sdf_path
            else:
                effective_sdf = source_sdf

            label = f"[{calc}] {wf_name} / {mol_key}"
            print(f"  Running {label} ...", flush=True)

            result = run_one(args, wf_name, wf, mol_key, mol_info,
                             effective_sdf, calc_root)
            all_results.append(result)

            status = f"  → {result['verdict']}  ({result['elapsed_s']}s)"
            if not result["passed"]:
                status += f"  [RC={result['returncode']}]"
            print(status, flush=True)

        print()

    # Reports
    json_path = args.output_root / "test_matrix_report.json"
    md_path = args.output_root / "test_matrix_report.md"
    write_json_report(json_path, all_results, args)
    write_markdown_report(md_path, all_results)

    n_pass = sum(1 for r in all_results if r["passed"])
    n_total = len(all_results)
    print(f"Results: {n_pass}/{n_total} passed")
    print(f"JSON  : {json_path}")
    print(f"Markdown: {md_path}")

    if args.benchmark and all_results:
        print("\nBenchmark — per-case timing:")
        print(f"  {'Case':<60} {'Time(s)':>8}")
        print(f"  {'-'*60} {'-'*8}")
        for r in sorted(all_results, key=lambda x: -x["elapsed_s"]):
            label = f"{r['calculator']}/{r['workflow']}/{r['molecule']}"
            print(f"  {label:<60} {r['elapsed_s']:>8.1f}")

    failed = [r for r in all_results if not r["passed"]]
    if failed:
        print(f"\nFailed cases ({len(failed)}):")
        for r in failed:
            print(f"  {r['case_id']}  →  {r['verdict']}  log: {r['log']}")

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
