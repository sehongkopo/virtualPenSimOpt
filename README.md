# virtualPenSimOpt

A live simulation of the 100,000 L penicillin fed-batch fermentation of IndPenSim, in which the same batch runs
side by side under three controllers of the study *Process optimisation of fed-batch fermentation using online
measurements alone* (Hong, submitted; preprint https://doi.org/10.26434/chemrxiv.15012001):

| Controller | What it reads | What it sets |
|---|---|---|
| No intervention | – | the recipe of the released code, with its PRBS perturbations of the feeds |
| APC | the true PAA concentration (as in the released code) | the PAA feed, by PI control at 1,200 mg/L from 10 h (Goldrick et al., 2019) |
| PPO | the sixteen on-line channels only | ten operating variables every 6 h from 12 h |

Every batch is integrated by [pyIndPenSim](https://github.com/sehongkopo/pyIndPenSim) with the BDF solver at the
tolerances of the study (relative 10⁻⁶, absolute 10⁻⁸). A batch is fixed by its seed, so the three controllers
meet the same initial conditions, strain parameters, disturbances and batch length. The dashboard streams every
0.2 h step to the browser while the batch runs.

## Run

```bash
git clone https://github.com/sehongkopo/virtualPenSimOpt.git
cd virtualPenSimOpt
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m vbr                     # opens http://127.0.0.1:8050
```

Python 3.9 or later is needed. PyTorch is not, because the policy is evaluated in NumPy. One batch takes about
10 s of computation per controller, and the dashboard can play it at 2, 5 or 10 h of batch time per second or as
fast as the computer allows. The batch seed can be any integer, and the 200 evaluation batches of the study are
offered in the seed list.

To serve the dashboard to other computers on the same network, bind it to all interfaces:

```bash
python -m vbr --host 0.0.0.0 --port 8060 --no-browser --max-runs 4
```

Each browser tab runs its own comparison. `--max-runs` caps how many run at once (each uses three processes), and a
comparison that no browser has watched for five minutes is stopped.

The dashboard shows
- a fermenter for the controller selected (no intervention, APC or PPO), whose broth level follows the volume and
  whose colour follows the biomass and turns murky when PAA is toxic, with the impeller, aeration and feeds running,
- dials for dissolved O₂, pH, PAA, sugar, biomass and penicillin, with a red blinking lamp when a variable is in a
  range that needs action (PAA above 2,400 mg/L, sugar above 10 g/L after 20 h, biomass or penicillin falling from
  its peak, DO below 5 mg/L, pH more than 0.25 from 6.5) and an amber lamp as it approaches,
- plots of the operating variables (sugar, PAA and oil feeds, aeration, base and coolant flows) and of the on-line
  observed variables (dissolved O₂, pH, temperature, off-gas CO₂, OUR, CER) for all three controllers,
- the net yield of each controller and every decision of the PPO policy, with a progress bar for the batch.

The alarm limits are display rules for this dashboard, not part of the study. PAA, sugar, biomass and penicillin
are not measured on-line and are not available to the PPO policy.

## Use from Python

```python
from vbr import run_batch

y, X = run_batch(847, 'ppo')              # net yield in kg and the pyIndPenSim batch structure
y_apc, _ = run_batch(847, 'apc')
```

`run_batch(seed, controller, policy_seed=0, on_step=None, on_decision=None)` accepts callbacks that receive every
step and every decision of the policy.

## Agreement with the study

`python validate.py` runs the 200 evaluation batches under the three controllers and compares the yields with the
reference values in `reference/evaluation_batches.npz`.

| Run | Mean yield here (kg) | Reference (kg) | Mean absolute difference per batch (kg) |
|---|---|---|---|
| No intervention | 2,799.5 | 2,799.5 | 0.0 |
| APC | 3,467.0 | 3,467.0 | 0.0 |
| PPO seed 0 | 3,626.6 | 3,591.7 | 115.5 |
| PPO seed 1 | 3,591.9 | 3,571.2 | 48.2 |
| PPO seed 2 | 3,666.1 | 3,686.6 | 44.4 |

No intervention and APC reproduce the reference batch by batch. The PPO reference was obtained by running the
policy in the GPU implementation used for training, with the carry-over of the sugar and PAA feeds of the
original code, and re-simulating its schedule of operating variables in pyIndPenSim. Here the policy reads the
channels of the pyIndPenSim batch itself, in closed loop, so its decisions differ slightly from batch to batch
while the means agree.

## Files

| File | Content |
|---|---|
| `vbr/engine.py` | runs one batch step by step under a controller |
| `vbr/policy.py` | the PPO network in NumPy and the construction of the sixteen-channel observation |
| `vbr/models/ppo_obs16_s{0,1,2}.npz` | weights of the PPO policies of seeds 0–2 |
| `vbr/server.py`, `vbr/static/index.html` | the local web dashboard (Python standard library and Plotly) |
| `validate.py` | comparison with the yields of the study |

## Licence

BSD 3-Clause (see `LICENSE`). IndPenSim is by Goldrick et al. (2015, 2019), and pyIndPenSim carries its own licence.

## Limitations

- The simulator is IndPenSim, not a plant. The PPO policy has been tested only in this simulation.
- APC acts on the true PAA concentration, as the released code does, and so represents an ideal Raman soft sensor.
- The dashboard has no authentication. Serve it only on a trusted network, not on the open internet.
