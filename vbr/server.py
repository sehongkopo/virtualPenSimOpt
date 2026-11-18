"""Local web dashboard for the virtual bioreactor (Python standard library only).

    python -m vbr [--port 8050] [--host 127.0.0.1]

The same batch is run under no intervention, APC and PPO in three processes, and every 0.2 h step
is streamed to the browser by server-sent events.
"""
import argparse, io, json, contextlib, threading, time, webbrowser
import multiprocessing as mp
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import numpy as np
from .engine import run_batch, CONTROLLERS, Stop

STATIC = Path(__file__).parent / 'static'
REF = Path(__file__).resolve().parent.parent / 'reference' / 'evaluation_batches.npz'


def _worker(seed, controller, q, speed, paused, stop):
    """Runs one batch and puts every step on the queue. speed = simulated hours per second (0 = as fast as possible)."""
    t0 = [time.time(), 0.0, speed.value]                          # anchor time, last batch time, speed in use

    def on_step(s, k, N):
        if stop.is_set():
            raise Stop
        while paused.is_set() and not stop.is_set():
            time.sleep(0.05); t0[2] = None                        # re-anchor after a pause
        if t0[2] != speed.value and speed.value > 0:              # speed changed or resumed: restart pacing from here
            t0[0] = time.time() - t0[1] / speed.value; t0[2] = speed.value
        if speed.value > 0:                                       # keep pace with the chosen speed
            due = t0[0] + s['t'] / speed.value; wait = due - time.time()
            if wait > 0: time.sleep(wait)
        t0[1] = s['t']
        q.put(('step', controller, s, k, N))

    def on_decision(dct):
        q.put(('decision', controller, dct, 0, 0))

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            y, _ = run_batch(seed, controller, on_step=on_step, on_decision=on_decision)
        q.put(('done', controller, {'yield': y}, 0, 0))
    except Stop:
        q.put(('stopped', controller, {}, 0, 0))
    except Exception as e:                                        # report a failed batch instead of hanging
        q.put(('error', controller, {'message': str(e)}, 0, 0))


class Run:
    """One live comparison of the three controllers on one seed."""
    def __init__(self, seed, speed):
        ctx = mp.get_context('spawn')
        self.seed = seed; self.events = []; self.lock = threading.Lock()
        self.q = ctx.Queue(); self.speed = ctx.Value('d', speed)
        self.paused = ctx.Event(); self.stop = ctx.Event()
        self.procs = [ctx.Process(target=_worker, args=(seed, c, self.q, self.speed, self.paused, self.stop), daemon=True)
                      for c in CONTROLLERS]
        for p in self.procs: p.start()
        threading.Thread(target=self._drain, daemon=True).start()
        self.seen = time.time()
        self._add({'type': 'start', 'seed': seed})

    def _add(self, ev):
        with self.lock: self.events.append(ev)

    def _drain(self):
        live = set(CONTROLLERS)
        while live:
            try:
                kind, c, payload, k, N = self.q.get(timeout=0.5)
            except Exception:
                if not any(p.is_alive() for p in self.procs): break
                continue
            ev = {'type': kind, 'c': c, **payload}
            if kind == 'step': ev |= {'k': k, 'N': N}
            self._add(ev)
            if kind in ('done', 'stopped', 'error'): live.discard(c)
        self._add({'type': 'finished'})

    def close(self):
        self.stop.set(); self.paused.clear()
        for p in self.procs: p.join(timeout=5)

    @property
    def active(self):
        return any(p.is_alive() for p in self.procs)


# One run per browser session, keyed by a random id that the page keeps in sessionStorage.
RUNS, RUNS_LOCK = {}, threading.Lock()
MAX_RUNS = 4          # concurrent comparisons (three processes each); set with --max-runs
IDLE_S = 300          # a run nobody has watched for this long is stopped and removed


def _reaper():
    while True:
        time.sleep(30)
        with RUNS_LOCK:
            old = [k for k, r in RUNS.items() if time.time() - r.seen > IDLE_S]
            gone = [RUNS.pop(k) for k in old]
        for r in gone: r.close()


def _sid(q):
    sid = q.get('sid', '')
    return sid if 8 <= len(sid) <= 64 and sid.replace('-', '').isalnum() else None


def seeds_info():
    if not REF.exists(): return []
    d = np.load(REF); yb = d['y_none']
    cls = np.where(yb < 2130, 'low', np.where(yb >= 3439, 'high', 'medium'))
    return [{'seed': int(s), 'class': str(c), 'hours': round(float(n) * 0.2, 1)} for s, c, n in zip(d['seeds'], cls, d['lens'])]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ('/', '/index.html'):
            b = (STATIC / 'index.html').read_bytes()
            self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)
        elif u.path == '/api/seeds':
            self._json(seeds_info())
        elif u.path == '/api/stream':
            self.send_response(200); self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache'); self.end_headers()
            sid = _sid({k: v[0] for k, v in parse_qs(u.query).items()})
            run, i = RUNS.get(sid), 0
            try:
                while run is not None and RUNS.get(sid) is run:
                    run.seen = time.time()
                    with run.lock: evs = run.events[i:]
                    i += len(evs)
                    if evs:
                        self.wfile.write(f'data: {json.dumps(evs)}\n\n'.encode()); self.wfile.flush()
                    if evs and evs[-1]['type'] == 'finished': break
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        sid = _sid(q)
        if sid is None:
            return self._json({'ok': False, 'message': 'missing session id'}, 400)
        run = RUNS.get(sid)
        if u.path == '/api/start':
            if run: run.close()
            with RUNS_LOCK:
                RUNS.pop(sid, None)
                busy = sum(r.active for r in RUNS.values())
                if busy >= MAX_RUNS:
                    return self._json({'ok': False, 'message': f'{busy} batches are running on the server. Try again in a minute.'}, 503)
                try:
                    seed = int(q.get('seed', 847)); speed = min(max(float(q.get('speed', 10)), 0.0), 1000.0)
                except ValueError:
                    return self._json({'ok': False, 'message': 'seed must be an integer'}, 400)
                RUNS[sid] = Run(seed, speed)
            self._json({'ok': True})
        elif u.path == '/api/pause' and run:
            (run.paused.clear if run.paused.is_set() else run.paused.set)(); self._json({'paused': run.paused.is_set()})
        elif u.path == '/api/speed' and run:
            run.speed.value = float(q.get('speed', 10)); self._json({'ok': True})
        elif u.path == '/api/stop' and run:
            run.close(); self._json({'ok': True})
        else:
            self._json({'ok': False}, 400)


def main():
    ap = argparse.ArgumentParser(description='Virtual bioreactor dashboard')
    ap.add_argument('--port', type=int, default=8050); ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--max-runs', type=int, default=MAX_RUNS, help='concurrent comparisons allowed')
    a = ap.parse_args()
    globals()['MAX_RUNS'] = a.max_runs
    threading.Thread(target=_reaper, daemon=True).start()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f'http://{a.host}:{a.port}/'
    print(f'Virtual bioreactor running at {url}  (Ctrl+C to stop)')
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for r in list(RUNS.values()): r.close()
