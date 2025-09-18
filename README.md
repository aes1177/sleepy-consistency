# Consistency in Sleepy Consensus using Markov Chains
This repository contains the main scripts used to derive and visualize Sleepy’s consistency conditions, estimate the finality threshold against the Balancing Attack, and run a lightweight simulator to empirically measure (in)consistency.

## Usage 
`python -m venv .venv`  
`source .venv/bin/activate`  
`pip install -r requirements.txt`  

`python conditions.py`  
> Interactive 2D plot of Sleepy consistency conditions

`python surfaces.py`  
> Interactive window with 3D surfaces of the Sleepy consistency conditions

`python balancing_attack.py`  
> Plots $P_0(k)$ and computes T-consistency thresholds

`python plot_heatmap.py --infile consistency_grid.csv --sigma 0.10 --out heatmap.png`  
> Creates the heatmap of the estimated experimental probabilities of the simulator. It requires the .csv file output of the SleepySimulator. We left a sample execution file `consistency_grid.csv`.

### SleepySimulator
The script must be called using the command `python -m dcsim.sleepy.sweep_consistency`.   
The simulator accepts a large number of parameters as input to customise the execution as much as possible. Below is a list of the main ones, which are also printed in the software `--help-args` command.

--------------------------------------------------------------------------------
    --help-args                    | flag     | default=False | Print a concise recap of all supported arguments and exit.
    --N                            | int      | default=100 | Total number of nodes.
    --sigma                        | float    | default=0.1 | Sleepy (offline) fraction σ.
    --delta                        | int      | default=5 | Maximum network delay bound Δ (in rounds).
    --k                            | int      | default=5 | Confirmation time k (safety parameter).
    --max-round                    | int      | default=1000 | Hard cap on rounds per run (applied after scaling).
    --rounds-baseline              | int      | default=200 | T0 baseline rounds; T = T0 * c / h (fallback to max-round or 200).
    --base-seed                    | int      | default=1337 | Base RNG seed (combined with c, σ, ρ_eff, rep).
    --reps                         | int      | default=50 | Independent repetitions per (c, ρ) point.
    --c-values                     | str      | default='' | Comma-separated list (e.g., 0.6,0.8,1.0,1.2).
    --c-min                        | float    | default=None | Minimum c (used with --c-max/--c-steps).
    --c-max                        | float    | default=None | Maximum c (used with --c-min/--c-steps).
    --c-steps                      | int      | default=None | Number of c points in the linear range (≥1).
    --rho-points                   | int      | default=10 | Fallback number of rho points per c (ignored if rho-step/density provided).
    --rho-step                     | float    | default=None | Fixed Δρ for all c (overrides --rho-points).
    --rho-density                  | float    | default=None | Points per ρ unit (equivalent to --rho-step=1/density).
    --rho-eps                      | float    | default=0.001 | Small margin ε to avoid boundary issues.
    --rho-split                    | str      | default='proportional' | Distribute points across bands proportionally to length or equally.
    --rho-start                    | str      | default='zero' | Start the rho grid from 'old*' threshold or from 0.
    --rho-max                      | float    | default=0.5 | Upper bound for requested ρ (clamped to <0.5).
    --delay-mode                   | str      | default='worst' | Adversarial worst-case delays or stochastic Beta(a,b) delays.
    --delay-beta                   | float    | default=[2.0, 5.0] | Parameters (a,b) for Beta in stochastic delay mode.
    --consistency-mode             | value    | default='both' | Common prefix only (cp), deep reorg only (reorg), or both.
    --consistency-stride           | int      | default=0 | Stride (rounds) between consistency checks (0 = every round).
    --outfile                      | str      | default='consistency_grid.csv' | CSV output path.
    --append                       | flag     | default=False | Append to CSV if it already exists.
    --cores                        | int      | default=1 | Number of worker processes (1 = serial).
--------------------------------------------------------------------------------

example call: 
`python -m dcsim.sleepy.sweep_consistency --N 100 --sigma 0.10 --delta 5 --k 3 --max-round 2000 --rounds-baseline 200 --reps 100 --c-min 0.5 --c-max 3 --c-steps 10 --rho-points 10 --rho-max 0.5 --delay-mode worst --cores 8 --outfile consistency_grid.csv`



