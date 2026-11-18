"""Runs one IndPenSim batch step by step under a chosen controller.

Controllers
  none  the recipe of the released code, with its PRBS perturbations of the feeds
  apc   PI control of PAA at 1,200 mg/L from 10 h (Goldrick et al., 2019), acting on the true PAA
  ppo   the reinforcement-learning policy, which reads only the sixteen on-line channels and
        sets ten operating variables every 6 h from 12 h

The batch is integrated by pyIndPenSim with the BDF solver at the tolerances of the study.
A callback receives every 0.2 h step, so a dashboard can follow the batch while it runs.
The same seed gives the same batch under every controller.
"""
from types import SimpleNamespace
import numpy as np
from pyindpensim.indpensim_run import prepare_batch
from pyindpensim.indpensim import indpensim
from pyindpensim.fctrl_indpensim import fctrl_indpensim
from .policy import Policy, ObsBuilder, LEVERS, DW, AK

CONTROLLERS = ('none', 'apc', 'ppo')


class Stop(Exception):
    """Raised inside the callback to abandon a running batch."""


def _flags(controller):
    f = SimpleNamespace(Control_strategy=[1], Batch_length=[1], Batch_fault_order_reference=[0], Raman_spec=[0])
    if controller == 'apc':
        f.PAA_setpoint_ctrl = True; f.PAA_sp = 1200.0; f.PAA_ctrl_start_h = 10.0
    return f


def _ph(v):
    """pyIndPenSim holds [H+] during the run and may convert it to pH afterwards."""
    return v if v > 1 else -np.log10(max(v, 1e-14))


def snapshot(X, k, h):
    """Values of the step just finished (index k-2), or the initial state for k = 1."""
    j = max(k - 2, 0)
    g = lambda ch: float(getattr(X, ch).y[j])
    P, V = g('P'), g('V')
    harvested = float(np.sum(X.Fremoved.y[:j + 1] * X.P.y[:j + 1]) * h) if k > 1 else 0.0
    return {
        't': round((k - 1) * h, 2),
        'DO2': g('DO2'), 'pH': _ph(g('pH')), 'T': g('T'),
        'CO2': g('CO2outgas'), 'OUR': g('OUR'), 'CER': g('CER'),
        'Fs': g('Fs'), 'Fpaa': g('Fpaa'), 'Fb': g('Fb'), 'Fc': g('Fc'),
        'PAA': g('PAA'), 'S': g('S'), 'X': g('a0') + g('a1') + g('a3') + g('a4'), 'P': P, 'V': V,
        'yield': (V * P - harvested) / 1000.0,
    }


def run_batch(seed, controller='ppo', policy_seed=0, on_step=None, on_decision=None):
    """Simulate one batch. Returns the net penicillin yield (kg) and the batch structure."""
    if controller not in CONTROLLERS:
        raise ValueError(f'controller must be one of {CONTROLLERS}')
    rng = np.random.default_rng(seed)
    Xinterp, x0, h, T, par, CF = prepare_batch(1, _flags(controller), rng)
    N = int(round(T / h))
    if controller == 'ppo':
        pol = Policy(policy_seed)
        lever = {m: np.ones(N) for m in LEVERS}; ph_sp = np.full(N, 6.5); t_sp = np.full(N, 298.0)
        CF.lever, CF.pH_sp, CF.T_sp = lever, ph_sp, t_sp
    state = {}

    def ctrl(X, Xd, k, h_, T_, CF_):
        i = k - 1
        if controller == 'ppo' and i % DW == 0:
            if 'obs' not in state:
                state['obs'] = ObsBuilder(X)
            o = state['obs'](X, i)
            dec = i // DW
            if dec >= AK:
                L = Policy.to_levers(pol(o))
                for m, name in enumerate(LEVERS):
                    lever[name][i:i + DW] = L[m]
                ph_sp[i:i + DW] = L[8]; t_sp[i:i + DW] = L[9]
                if on_decision:
                    on_decision({'t': round(i * h_, 1), **{n: float(L[m]) for m, n in enumerate(LEVERS)},
                                 'pH_sp': float(L[8]), 'T_sp': float(L[9])})
        u, X = fctrl_indpensim(X, Xd, k, h_, T_, CF_)
        if on_step and k > 1:
            on_step(snapshot(X, k, h_), k, N)
        return u, X

    X = indpensim(ctrl, Xinterp, x0, h, T, 2, par, CF, rng=rng, rtol=1e-6, atol=1e-8)
    y = float((X.V.y[-1] * X.P.y[-1] - np.sum(X.Fremoved.y * X.P.y) * h) / 1000.0)
    if on_step:
        on_step(snapshot(X, N + 1, h) | {'yield': y}, N + 1, N)
    return y, X
