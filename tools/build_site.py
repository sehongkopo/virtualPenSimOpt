"""Builds the static replay site for GitHub Pages.

    python tools/build_site.py [--out site] [--workers 32] [--seeds 847 1 2 ...]

Every evaluation batch of the study is run under no intervention, APC and PPO, and every 0.2 h step
is stored, so the dashboard can replay the batches in a browser with no Python server. Each seed
gives two files, data/<seed>.json (batch length, yields, PPO decisions and the scale of each variable)
and data/<seed>.bin (every variable of every step as unsigned 16-bit integers on that scale).
The page is vbr/static/index.html with the replay switched on.
"""
import argparse, contextlib, io, json, os, shutil, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CTRLS = ('none', 'apc', 'ppo')
VARS = ('DO2', 'pH', 'T', 'CO2', 'OUR', 'CER', 'Fs', 'Fpaa', 'Fb', 'Fc', 'Foil', 'Fg', 'RPM', 'Fremoved',
        'PAA', 'S', 'X', 'P', 'V', 'yield')


def one(job):
    seed, c = job
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from vbr.engine import run_batch
    steps, decs = [], []
    with contextlib.redirect_stdout(io.StringIO()):
        y, _ = run_batch(seed, c, on_step=lambda s, k, N: steps.append((k, N, s)), on_decision=decs.append)
    return seed, c, y, steps, decs


def pack(seed, runs, out):
    """runs[c] = (yield, steps, decisions). Writes data/<seed>.json and data/<seed>.bin."""
    head = {'seed': seed, 'vars': list(VARS), 'ctrls': {}}
    blobs = []
    for c in CTRLS:
        y, steps, decs = runs[c]
        k = np.array([s[0] for s in steps], dtype=np.uint16)
        M = np.array([[s[2][v] for v in VARS] for s in steps], dtype=np.float64)   # steps x vars
        lo, hi = M.min(0), M.max(0)
        span = np.where(hi > lo, hi - lo, 1.0)
        q = np.round((M - lo) / span * 65535).astype(np.uint16)
        head['ctrls'][c] = {'n': len(steps), 'N': int(steps[0][1]), 'yield': y, 'lo': lo.tolist(), 'hi': hi.tolist(),
                            'decisions': decs}
        blobs += [k.tobytes(), np.ascontiguousarray(q.T).tobytes()]               # k, then variable by variable
    (out / f'{seed}.json').write_text(json.dumps(head, separators=(',', ':')))
    (out / f'{seed}.bin').write_bytes(b''.join(blobs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'site'))
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--seeds', type=int, nargs='*')
    a = ap.parse_args()
    ref = np.load(ROOT / 'reference' / 'evaluation_batches.npz')
    seeds = a.seeds or [int(s) for s in ref['seeds']]
    out = Path(a.out); data = out / 'data'; data.mkdir(parents=True, exist_ok=True)

    page = (ROOT / 'vbr' / 'static' / 'index.html').read_text()
    page = page.replace('<script>', '<script>window.VBR_STATIC = true;</script>\n<script>', 1)
    (out / 'index.html').write_text(page)
    (out / '.nojekyll').write_text('')

    yb = ref['y_none']; cls = np.where(yb < 2130, 'low', np.where(yb >= 3439, 'high', 'medium'))
    info = {int(s): {'class': str(c), 'hours': round(float(n) * 0.2, 1)} for s, c, n in zip(ref['seeds'], cls, ref['lens'])}

    todo = [s for s in seeds if not ((data / f'{s}.json').exists() and (data / f'{s}.bin').exists())]
    runs, done = {}, 0
    with Pool(a.workers) as pool:
        for seed, c, y, steps, decs in pool.imap_unordered(one, [(s, c) for s in todo for c in CTRLS]):
            runs.setdefault(seed, {})[c] = (y, steps, decs)
            if len(runs[seed]) == len(CTRLS):
                pack(seed, runs.pop(seed), data); done += 1
                print(f'{done}/{len(todo)} seed {seed}', flush=True)

    listing, yields = [], {c: [] for c in CTRLS}
    for s in seeds:
        h = json.loads((data / f'{s}.json').read_text())
        for c in CTRLS: yields[c].append(h['ctrls'][c]['yield'])
        listing.append({'seed': s, **info.get(s, {'class': 'other', 'hours': round(h['ctrls']['none']['N'] * 0.2, 1)})})
    (data / 'seeds.json').write_text(json.dumps(listing))
    print('mean yield (kg):', {c: round(float(np.mean(v)), 1) for c, v in yields.items()})


if __name__ == '__main__':
    main()
