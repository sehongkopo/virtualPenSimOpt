"""Checks the virtual bioreactor against the evaluation of the study.

Runs the 200 evaluation batches of the study (reference/evaluation_batches.npz) under no intervention, APC and
the PPO policies of seeds 0-2, and compares the yields with the reference values.

    python validate.py [n_batches] [workers]

No intervention and APC should agree to rounding. The PPO reference was obtained by running the policy in the
GPU implementation used for training and re-simulating its schedule of operating variables in pyIndPenSim.
The closed loop run here lets the policy read the pyIndPenSim batch itself, so single batches differ slightly
while the means agree.
"""
import os, sys, io, contextlib
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from vbr import run_batch

REF = Path(__file__).resolve().parent / 'reference' / 'evaluation_batches.npz'


def one(job):
    seed, c, ps = job
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            return run_batch(seed, c, policy_seed=ps)[0]
        except Exception:
            return float('nan')


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    w = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    d = np.load(REF)
    seeds = [int(s) for s in d['seeds'][:n]]
    runs = [('none', 0), ('apc', 0), ('ppo', 0), ('ppo', 1), ('ppo', 2)]
    jobs = [(s, c, p) for c, p in runs for s in seeds]
    with ProcessPoolExecutor(max_workers=w) as ex:
        y = np.array(list(ex.map(one, jobs))).reshape(len(runs), n)
    print(f'{n} evaluation batches, mean net yield (kg)')
    print(f'{"run":10s} {"here":>8s} {"reference":>10s} {"mean |diff|":>12s}   reference')
    for (c, p), r in zip(runs, y):
        refs = [('y_none', d['y_none'])] if c == 'none' else [('y_apc', d['y_apc'])] if c == 'apc' else \
               [('y_ppo', d['y_ppo'][p])]
        for name, ref in refs:
            ref = ref[:n]
            print(f'{c + (f" s{p}" if c == "ppo" else ""):10s} {np.nanmean(r):8.1f} {np.nanmean(ref):10.1f} '
                  f'{np.nanmean(np.abs(r - ref)):12.1f}   {name}')
    np.savez('validate_result.npz', seeds=seeds, runs=np.array([f'{c}{p}' for c, p in runs]), y=y)
