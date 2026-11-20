"""Model worker for DeepConf external calculators (UMA, MACE-OMol, AIMNet2-2025).

DeepConf starts this script with the Python interpreter of the environment that holds the model
(see confGen.setExternalCalculator in core_conf.py). The worker loads the model once and then serves
batched energies and forces over a private unix socket, so that DeepConf's own batched FIRE optimizer
drives every model in the same way.

It needs only numpy, torch, ase and the model's own package; DeepConf itself need not be installed
in the worker environment.

Units: energies in eV, forces in eV/Angstrom.
The connection key is read from the environment variable DEEPCONF_WORKER_AUTHKEY (hex string).
"""
import argparse
import os
import warnings
from multiprocessing.connection import Listener

import numpy as np

warnings.filterwarnings("ignore")

KINDS = ("aimnet", "uma", "mace")


def _pick_device(requested):
    import torch
    requested = (requested or "auto").lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _ase_evaluator(calc, charge, mult):
    """One ASE call per structure; total charge and spin multiplicity go in atoms.info."""
    from ase import Atoms

    def evaluate(pos, numbers, forces):
        energies, all_forces = [], []
        for x in pos:
            atoms = Atoms(numbers=numbers, positions=np.asarray(x, dtype=np.float64))
            atoms.info.update(charge=charge, spin=mult)
            atoms.calc = calc
            energies.append(atoms.get_potential_energy())
            if forces:
                all_forces.append(atoms.get_forces())
        return (np.array(energies, dtype=np.float64),
                np.array(all_forces, dtype=np.float64) if forces else None)

    return evaluate


def build_evaluator(kind, model, device, charge, mult, task, dtype):
    """Load the model and return (evaluate(pos[B,N,3], numbers[N], forces) -> (E[B], F[B,N,3] or None), label)."""
    if kind == "aimnet":
        import torch
        from aimnet.calculators import AIMNet2Calculator
        model = model or "aimnet2-2025"
        calc = AIMNet2Calculator(model, device=device)

        def evaluate(pos, numbers, forces):
            nbatch = pos.shape[0]
            data = {
                "coord": torch.tensor(pos, dtype=torch.float32),
                "numbers": torch.tensor(numbers, dtype=torch.int64)[None].expand(nbatch, -1),
                "charge": torch.full((nbatch,), float(charge)),
                "mult": torch.full((nbatch,), float(mult)),
            }
            res = calc.eval(data, forces=forces)
            energy = res["energy"].detach().cpu().numpy().reshape(nbatch).astype(np.float64)
            force = (res["forces"].detach().cpu().numpy().reshape(nbatch, -1, 3).astype(np.float64)
                     if forces else None)
            return energy, force

        return evaluate, f"AIMNet2 model {model}"

    if kind == "uma":
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        model = model or "uma-s-1p2"
        task = task or "omol"
        if os.path.exists(os.path.expanduser(model)):
            unit = pretrained_mlip.load_predict_unit(os.path.expanduser(model), device=device)
        else:
            unit = pretrained_mlip.get_predict_unit(model, device=device)
        calc = FAIRChemCalculator(unit, task_name=task)
        return _ase_evaluator(calc, charge, mult), f"UMA model {model}, task {task}"

    if kind == "mace":
        from mace.calculators import mace_omol
        dtype = dtype or "float64"
        calc = mace_omol(os.path.expanduser(model) if model else None, device=device, default_dtype=dtype)
        return _ase_evaluator(calc, charge, mult), f"MACE-OMol model {model or '(package default)'}, {dtype}"

    raise ValueError(f"unknown model kind {kind!r}; expected one of {KINDS}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", required=True, choices=KINDS)
    parser.add_argument("--socket", required=True, help="path of the unix socket to listen on")
    parser.add_argument("--model", default="", help="model name or file (blank = default of the model package)")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--mult", type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto, cuda or cpu")
    parser.add_argument("--task", default="", help="UMA task name (default omol)")
    parser.add_argument("--dtype", default="", help="MACE default dtype (default float64)")
    args = parser.parse_args()

    key_hex = os.environ.pop("DEEPCONF_WORKER_AUTHKEY", "")
    if not key_hex:
        raise SystemExit("ext_worker: DEEPCONF_WORKER_AUTHKEY is not set (this script is started by DeepConf)")
    authkey = bytes.fromhex(key_hex)

    device = _pick_device(args.device)
    evaluate, label = build_evaluator(args.kind, args.model, device, args.charge, args.mult, args.task, args.dtype)
    print(f"[ext_worker] {label} on {device}, charge={args.charge}, mult={args.mult}", flush=True)

    # The socket appears only now, after the model is loaded: DeepConf takes it as the ready signal.
    with Listener(args.socket, family="AF_UNIX", authkey=authkey) as listener:
        with listener.accept() as conn:
            conn.send(("ready", device))
            while True:
                try:
                    msg = conn.recv()
                except EOFError:      # DeepConf has gone away
                    break
                if msg[0] == "stop":
                    break
                _, pos, numbers, forces = msg
                try:
                    conn.send(("ok",) + evaluate(np.asarray(pos), np.asarray(numbers), forces))
                except Exception as exc:
                    conn.send(("err", repr(exc)))


if __name__ == "__main__":
    main()
