"""PPO policy of the study, evaluated in NumPy.

The network reads the sixteen on-line channels and returns the mean of the ten actions.
Weights were exported from the trained PyTorch policies (seeds 0-2, narrow amplitude,
start 10 h, decisions every 6 h), so PyTorch is not needed to run the policy.
"""
from pathlib import Path
import numpy as np

MODEL_DIR = Path(__file__).parent / 'models'

# Operating variables in the order of the policy output (eight multipliers, then two set-points)
LEVERS = ['Fs', 'Fpaa', 'Foil', 'Fg', 'Fw', 'Fremoved', 'RPM', 'pressure']
LO = np.array([0.985, 0.985, 0.96, 0.96, 0.92, 0.92, 0.97, 0.97, 6.35, 296.5])
HI = np.array([1.001, 1.001, 1.04, 1.04, 1.08, 1.08, 1.03, 1.03, 6.65, 299.5])
BASE = np.array([1, 1, 1, 1, 1, 1, 1, 1, 6.5, 298.0])
HALF = np.maximum(BASE - LO, HI - BASE)

DW = 30            # steps of 0.2 h between decisions (6 h)
AK = 2             # first acting decision (12 h, the first decision at or after 10 h)
NMAX = 1560        # steps of the longest batch in the training bank, used for the batch-time channel

# The sixteen channels: name, unit, and how the policy scales them
CHANNELS = [
    ('Dissolved O2', 'mg/L'), ('[H+] (pH)', 'mol/L'), ('Temperature', 'K'), ('Off-gas CO2', '%'),
    ('Off-gas O2', 'mol frac'), ('Vessel weight', 'kg'), ('Volume', 'L'),
    ('dDO2 over 6 h', '-'), ('dCO2 over 6 h', '-'), ('Sugar feed Fs', 'L/h'), ('PAA feed Fpaa', 'L/h'),
    ('Batch time', '-'), ('OUR', 'g/min'), ('CER', 'g/h'), ('Base flow Fb', 'L/h'), ('Coolant flow Fc', 'L/h'),
]


class Policy:
    def __init__(self, seed=0):
        w = np.load(MODEL_DIR / f'ppo_obs16_s{seed}.npz')
        self.W1, self.b1, self.W2, self.b2 = w['W1'], w['b1'], w['W2'], w['b2']
        self.Wmu, self.bmu = w['Wmu'], w['bmu']

    def __call__(self, obs):
        h = np.tanh(self.W1 @ obs + self.b1)
        h = np.tanh(self.W2 @ h + self.b2)
        return self.Wmu @ h + self.bmu

    @staticmethod
    def to_levers(a):
        """Action in [-1, 1] -> eight multipliers, pH set-point and temperature set-point."""
        return np.clip(BASE + np.clip(a, -1, 1) * HALF, LO, HI)


class ObsBuilder:
    """Builds the sixteen-channel observation from a running pyIndPenSim batch.

    Matches the GPU environment used for training: the latest state, the feeds applied in the
    previous 0.2 h step, the changes in dissolved O2 and off-gas CO2 since the previous decision,
    and OUR and CER computed from the off-gas balance.
    """
    def __init__(self, X):
        self.do_prev = float(X.DO2.y[0]); self.co2_prev = float(X.CO2outgas.y[0])

    def __call__(self, X, i):
        j = max(i - 1, 0)                 # latest stored state (after i steps)
        s = lambda ch: float(getattr(X, ch).y[j])
        DO, CO2 = s('DO2'), s('CO2outgas')
        fo = [DO / 15, s('pH') / 5e-8, s('T') / 300, CO2 / 2, s('O2') / 0.21, s('Wt') / 1e5, s('V') / 1e5]
        dDO = np.clip((DO - self.do_prev) / 2.0, -2, 2); dCO2 = np.clip((CO2 - self.co2_prev) / 0.1, -2, 2)
        self.do_prev, self.co2_prev = DO, CO2
        fs, fpaa, fg, fb, fc = (s(c) for c in ('Fs', 'Fpaa', 'Fg', 'Fb', 'Fc'))
        o2 = np.clip(s('O2'), 0, 0.25); co = np.clip(CO2, 0, 20)
        our = (32 * fg / 22.4) * (0.204 - o2 * (0.7902 / max(1 - o2 - co / 100, 0.05)))
        cer = (44 * fg / 22.4) * ((0.65 * co / 100) * (0.7902 / max(1 - 0.204 - co / 100, 0.05) - 0.0330))
        tfrac = (i // DW) * DW / NMAX
        ex = [np.clip(our / 2, -3, 3), np.clip(cer / 2, -3, 3), np.clip(fb / 100, -3, 3), np.clip(fc / 500, -3, 3)]
        return np.array(fo + [dDO, dCO2, np.clip(fs / 100, 0, 3), np.clip(fpaa / 8, 0, 3), tfrac] + ex)
