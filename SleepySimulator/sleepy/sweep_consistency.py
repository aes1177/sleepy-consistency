# sweep_consistency.py
# ------------------------------------------------------------------------------
# Purpose: sweep over a grid of (c, rho) points and estimate the probability
#          of breaking Sleepy protocol consistency (default attack: Balancing Attack),
#          saving results to CSV.
#          Includes:
#            - flexible rho-grid construction (starting from 0 or from the "old" threshold),
#            - dynamic round budget T = T0 * c / h,
#            - p-hat + 95% CI (Wilson),
#            - process-level parallelism
# ------------------------------------------------------------------------------

import math
import csv
import argparse
import os
import sys
import hashlib
import time
import textwrap
from typing import Optional, Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import brentq

# --- minimal WARNING-level logging
import logging
root = logging.getLogger()
root.setLevel(logging.WARNING)
for h in list(root.handlers):
    root.removeHandler(h)

# --- dcsim imports: runner, honest nodes, measurements, attacks, config, fast signing
from dcsim.framework import Runner
from dcsim.sleepy.HonestNode import HonestNode
from dcsim.sleepy.ConsistencyMeasurement import ConsistencyMeasurement
from dcsim.sleepy.BalancingAttack import BalancingAttack
from dcsim.sleepy.Configuration import Configuration
from dcsim.sleepy.FSignFast import FSignFast as FSign

# =========================
# 95% Wilson confidence interval for proportions
# =========================
def ci95_wilson(phat: float, n: int) -> Tuple[float, float]:
    """
    95% confidence interval for a binomial proportion using the Wilson score interval.
    Robust even for small n and p̂ near 0/1.
    """
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt((phat * (1 - phat) / n) + (z * z) / (4 * n * n))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi

# =========================
# Theoretical conditions (old/new)
# =========================
def cond_old(c: float, sigma: float, rho: float) -> bool:
    """
    Old condition: (1 - 2h/c) * h - rho > 0 with h = 1 - sigma - rho.
    """
    h = 1 - sigma - rho
    return (h > 0 and c > 0 and ((1 - 2 * h / c) * h - rho) > 0)

def cond_new(c: float, sigma: float, rho: float) -> bool:
    """
    New condition: h * exp(-2h/c) - rho > 0 with h = 1 - sigma - rho.
    """
    h = 1 - sigma - rho
    return (h > 0 and c > 0 and (h * math.exp(-2 * h / c) - rho) > 0)

# =========================
# Robust root finding (scan + Brent)
# =========================
def _root_with_scan(f, lo: float, hi: float, steps: int = 400) -> Optional[float]:
    """
    Scan [lo, hi] in 'steps' increments to detect a sign change, then refine with brentq.
    Returns None if no bracketing is found.
    """
    if not (lo < hi):
        return None
    lo = max(lo, 1e-12)
    hi = max(lo + 1e-12, hi)
    x_prev = lo
    f_prev = f(x_prev)
    for i in range(1, steps + 1):
        x = lo + (hi - lo) * i / steps
        fx = f(x)
        if f_prev == 0.0:
            return x_prev
        if fx == 0.0:
            return x
        if (f_prev < 0 < fx) or (f_prev > 0 > fx):
            try:
                return brentq(f, x_prev, x)
            except ValueError:
                pass
        x_prev, f_prev = x, fx
    return None

def solve_thresholds(c: float, sigma: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute (rho_old_star, rho_new_star) by solving f_old=0 and f_new=0,
    if roots exist in (0, 1 - sigma).
    """
    if c <= 0 or sigma >= 1:
        return None, None
    eps = 1e-6
    lo, hi = eps, max(eps, 1 - sigma - eps)

    def f_new(rho: float) -> float:
        h = 1 - sigma - rho
        return -1.0 if h <= 0 else (h * math.exp(-2 * h / c) - rho)

    def f_old(rho: float) -> float:
        h = 1 - sigma - rho
        return -1.0 if h <= 0 else ((1 - 2 * h / c) * h - rho)

    return _root_with_scan(f_old, lo, hi), _root_with_scan(f_new, lo, hi)

# =========================
# Stable (deterministic) seed from items
# =========================
def stable_seed(*items) -> int:
    """
    Concatenate items into a string and compute a BLAKE2b digest (8 bytes),
    converted to an integer. Used to get reproducible seeds for (c, sigma, rho_eff, rep).
    """
    s = "|".join(map(repr, items)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(s, digest_size=8).digest(), "big")

# =========================
# Build rho grid for a single c
# =========================
def build_rho_grid(
    c: float,
    sigma: float,
    n_points: int,
    eps: float = 1e-3,
    split: str = "proportional",
    rho_max: float = 0.5,
    rho_step: Optional[float] = None,
    rho_density: Optional[float] = None,
    rho_start: str = "old",  # "old": start from old*; "zero": start from 0
):
    """
    Return (rho_list, rho_old_star, rho_new_star).

    Options:
      - rho_start="zero": single band [ε, min(1-σ, ρ_max)-ε]
      - rho_start="old" : start just above old* if it exists, else ε; if new* exists, split into two bands
      - rho_step        : fixed Δρ (overrides n_points)
      - rho_density     : points per unit (equivalent to Δρ = 1/ρ_density)
    """
    rho_old_star, rho_new_star = solve_thresholds(c, sigma)

    # do not exceed 1 - sigma nor rho_max; avoid borders by eps
    hard_upper = min(1.0 - sigma - eps, rho_max - eps)
    upper = max(eps, hard_upper)

    segs = []
    if rho_start == "zero":
        # cover the whole interval from 0 (ε) up to rho_max/1-σ
        if upper > eps:
            segs.append(("all", eps, upper))
    else:
        # start from the old* threshold if it exists, otherwise from ε
        start_from = (rho_old_star + eps) if (rho_old_star is not None) else eps
        if rho_new_star is not None:
            # [old*+ε, new*-ε] and [new*+ε, upper]
            Lb = start_from
            Hb = min(rho_new_star - eps, hard_upper)
            if Hb > Lb:
                segs.append(("between", Lb, Hb))
            Lh, Hh = rho_new_star + eps, hard_upper
            if Hh > Lh:
                segs.append(("high", Lh, Hh))
        else:
            # if no "new" threshold, a single band [start_from, hard_upper]
            if hard_upper > start_from:
                segs.append(("from_old", start_from, hard_upper))

    if not segs:
        return [], rho_old_star, rho_new_star

    # --- constant-density mode: fixed Δρ via rho_step or rho_density (overrides n_points)
    step = None
    if rho_step is not None and rho_step > 0:
        step = float(rho_step)
    elif rho_density is not None and rho_density > 0:
        step = 1.0 / float(rho_density)

    if step is not None:
        grid: List[float] = []
        for _, L, H in segs:
            if H - L <= step * 0.999:
                # very short interval: take the midpoint
                grid.append((L + H) / 2.0)
            else:
                # generate L, L+step, ..., including H (without overshooting)
                x = L
                while x < H - 1e-12:
                    grid.append(x)
                    x += step
                grid.append(H)
        # deduplicate + round to 6 decimals for numerical/CSV stability
        grid = sorted({round(x, 6) for x in grid if eps <= x <= hard_upper})
        return grid, rho_old_star, rho_new_star

    # --- fallback: split n_points across bands (proportional or equal)
    n_points = max(1, int(n_points))
    if len(segs) == 1:
        counts = [n_points]
    else:
        if split == "proportional":
            lengths = [H - L for _, L, H in segs]
            base = [1] * len(segs)               # ensure ≥ 1 point per band
            rem = n_points - sum(base)
            tot = sum(lengths)
            alloc = [int(round(rem * (l / tot))) for l in lengths] if tot > 0 else [rem // len(segs)] * len(segs)
            # fix rounding differences
            diff = rem - sum(alloc)
            i = 0
            while diff != 0:
                j = i % len(alloc)
                if diff > 0:
                    alloc[j] += 1
                    diff -= 1
                elif alloc[j] > 0:
                    alloc[j] -= 1
                    diff += 1
                i += 1
            counts = [b + a for b, a in zip(base, alloc)]
        else:
            # split "equal": same count per band (remainder to the last band)
            a = n_points // len(segs)
            counts = [a] * (len(segs) - 1) + [n_points - a * (len(segs) - 1)]

    # generate evenly spaced points in each band
    grid = []
    for (_, L, H), m in zip(segs, counts):
        if m <= 1:
            grid.append((L + H) / 2.0)
        else:
            st = (H - L) / (m - 1)
            grid.extend([L + i * st for i in range(m)])

    # deduplicate + final clamp
    grid = sorted({round(x, 6) for x in grid if eps <= x <= hard_upper})
    return grid, rho_old_star, rho_new_star

# =========================
# Single (c, rho) point → Monte Carlo estimate
# =========================
def eval_point(c: float, rho_req: float, args) -> Optional[dict]:
    """
    Run 'reps' independent simulations at (c, rho_req) and return summary stats
    (p̂, CI95, counters, effective parameters). Returns None if h <= 0 (no honest activity).
    """
    # --- global simulation parameters
    N = args.N
    sigma = args.sigma
    delta = args.delta
    k = args.k
    max_round_cap = args.max_round
    T0 = args.rounds_baseline
    base_seed = args.base_seed

    # --- clamp requested rho to [ε, 0.5)
    eps = 1e-6
    rho_req = max(eps, min(0.5 - eps, rho_req))

    # --- discretization: integer corrupted count → effective rho
    num_corrupted = max(1, min(int(round(N * rho_req)), int(N * 0.5)))
    num_honest = N - num_corrupted
    rho_eff = num_corrupted / N

    # --- h = active honest fraction; if non-positive, the point is invalid
    h = 1.0 - sigma - rho_eff
    if h <= 0:
        return None

    # --- probability of electing a leader (honest/corrupt) coherent with honest rate
    p = 1.0 / (delta * N * c)

    # --- dynamic round budget T: scales with c/h and respects k and max_round
    if T0 is None or T0 <= 0:
        # default: if not provided, use cap max_round (if >0) or fallback to 200
        T0 = max_round_cap if (max_round_cap and max_round_cap > 0) else 200
    T = int(math.ceil(T0 * c / max(h, 1e-12)))  # T = T0 * c / h
    T = max(T, k + 1)                           # at least 1 commit-eligible block
    if max_round_cap is not None and max_round_cap > 0:
        T = min(T, int(max_round_cap))          # global safety cap

    # --- Monte Carlo across 'reps'
    successes = 0
    for rep in range(args.reps):
        seed = stable_seed(base_seed, c, sigma, rho_eff, rep)

        # Configuration: Balancing Attack + standard consistency measurement
        cfg = Configuration(
            honest_node_type=HonestNode,
            adversary_controller_type=BalancingAttack,
            measurement_type=ConsistencyMeasurement,
            num_honest_nodes=num_honest,
            num_corrupted_nodes=num_corrupted,
            max_delay=delta,
            confirm_time=k,
            probability=p,
            max_round=T,                 # dynamic round budget
            seed=seed,
            sigma=sigma,
            c=c,
            compact_broadcast=True,
            auth_checks=False,
            enable_txs=False,
            delay_mode=args.delay_mode,
            delay_beta_a=float(args.delay_beta[0]),
            delay_beta_b=float(args.delay_beta[1]),
        )

        # Consistency measurement params
        cfg.consistency_mode = str(args.consistency_mode).lower()  # cp/reorg/both
        cfg.consistency_stride = max(0, int(args.consistency_stride))

        # Runner: add TTP fast-sign, init, run; ok=True if inconsistency detected
        runner = Runner(cfg)
        runner.add_trusted_third_party(FSign("FSign"))
        runner.init()

        ok = runner.run()
        if ok:
            successes += 1

    # --- estimate p̂ and 95% CI
    phat = successes / args.reps
    lo, hi = ci95_wilson(phat, args.reps)

    # --- package results for CSV
    return {
        "c": c,
        "rho_requested": rho_req,
        "rho_effective": rho_eff,
        "N": N,
        "sigma": sigma,
        "delta": delta,
        "k": k,
        "max_round": T,
        "rounds_baseline": T0,
        "reps": args.reps,
        "p": p,
        "num_honest": num_honest,
        "num_corrupted": num_corrupted,
        "successes": successes,
        "p_hat": phat,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "cond_old": cond_old(c, sigma, rho_req),
        "cond_new": cond_new(c, sigma, rho_req),
    }

# =========================
# CLI: arguments and options
# =========================
def _print_args_recap(parser: argparse.ArgumentParser):
    """
    Print a compact recap of all supported CLI arguments (flags, type, default, help).
    """
    print("\nArguments recap")
    print("-" * 80)
    for act in parser._actions:
        # skip positionals with no option strings
        if not act.option_strings:
            continue
        # skip the built-in -h/--help to keep recap concise (optional)
        if any(s in ("-h", "--help") for s in act.option_strings):
            continue
        opt = ", ".join(act.option_strings)
        # infer a readable type
        if getattr(act, "nargs", None) in ("*", "+", argparse.REMAINDER):
            typ = "list"
        elif hasattr(act, "type") and act.type is not None:
            typ = getattr(act.type, "__name__", str(act.type))
        elif isinstance(act, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            typ = "flag"
        else:
            typ = "value"
        default = act.default
        helpmsg = (act.help or "").strip()
        print(f"{opt:30} | {typ:8} | default={default!r} | {helpmsg}")
    print("-" * 80)
    print("Tip: run with --help for full, grouped help with examples.\n")

def parse_args():
    """
    Define all CLI arguments to configure the sweep.
    """
    ap = argparse.ArgumentParser(
        description="Sweep over (c, rho) to estimate probability of breaking Sleepy consistency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              # Quick smoke test on a few c's, serial execution, small reps
              python sweep_consistency.py --c-values 0.6,1.0,1.5 --rho-points 8 --reps 20 --outfile out.csv

              # Generate c-range, start rho grid from old*, fixed rho step, 4 cores
              python sweep_consistency.py --c-min 0.6 --c-max 2.0 --c-steps 8 \\
                  --rho-start old --rho-step 0.02 --cores 4

              # Use stochastic delays with Beta(a,b) and higher baseline rounds
              python sweep_consistency.py --delay-mode stochastic --delay-beta 2 8 --rounds-baseline 400
            """
        ),
    )

    # top-level quick recap flag
    ap.add_argument(
        "--help-args",
        action="store_true",
        help="Print a concise recap of all supported arguments and exit.",
    )

    # Protocol/System parameters
    g_sys = ap.add_argument_group("Protocol/System")
    g_sys.add_argument("--N", type=int, default=100, help="Total number of nodes.")
    g_sys.add_argument("--sigma", type=float, default=0.10, help="Sleepy (offline) fraction σ.")
    g_sys.add_argument("--delta", type=int, default=5, help="Maximum network delay bound Δ (in rounds).")
    g_sys.add_argument("--k", type=int, default=5, help="Confirmation time k (safety parameter).")
    g_sys.add_argument("--max-round", type=int, default=1000, help="Hard cap on rounds per run (applied after scaling).")
    g_sys.add_argument("--rounds-baseline", type=int, default=200, help="T0 baseline rounds; T = T0 * c / h (fallback to max-round or 200).")
    g_sys.add_argument("--base-seed", type=int, default=1337, help="Base RNG seed (combined with c, σ, ρ_eff, rep).")
    g_sys.add_argument("--reps", type=int, default=50, help="Independent repetitions per (c, ρ) point.")

    # c-grid construction
    g_c = ap.add_argument_group("c-grid")
    g_c.add_argument("--c-values", type=str, default="", help="Comma-separated list (e.g., 0.6,0.8,1.0,1.2).")
    g_c.add_argument("--c-min", type=float, default=None, help="Minimum c (used with --c-max/--c-steps).")
    g_c.add_argument("--c-max", type=float, default=None, help="Maximum c (used with --c-min/--c-steps).")
    g_c.add_argument("--c-steps", type=int, default=None, help="Number of c points in the linear range (≥1).")

    # rho-grid construction
    g_rho = ap.add_argument_group("rho-grid")
    g_rho.add_argument("--rho-points", type=int, default=10, help="Fallback number of rho points per c (ignored if rho-step/density provided).")
    g_rho.add_argument("--rho-step", type=float, default=None, help="Fixed Δρ for all c (overrides --rho-points).")
    g_rho.add_argument("--rho-density", type=float, default=None, help="Points per ρ unit (equivalent to --rho-step=1/density).")
    g_rho.add_argument("--rho-eps", type=float, default=1e-3, help="Small margin ε to avoid boundary issues.")
    g_rho.add_argument("--rho-split", type=str, default="proportional", choices=["proportional", "equal"], help="Distribute points across bands proportionally to length or equally.")
    g_rho.add_argument("--rho-start", type=str, default="zero", choices=["old", "zero"], help="Start the rho grid from 'old*' threshold or from 0.")
    g_rho.add_argument("--rho-max", type=float, default=0.5, help="Upper bound for requested ρ (clamped to <0.5).")

    # Network / delays
    g_net = ap.add_argument_group("Network/Delays")
    g_net.add_argument("--delay-mode", type=str, default="worst", choices=["worst", "stochastic"], help="Adversarial worst-case delays or stochastic Beta(a,b) delays.")
    g_net.add_argument("--delay-beta", type=float, nargs=2, default=[2.0, 5.0], help="Parameters (a,b) for Beta in stochastic delay mode.")

    # Consistency measurement
    g_cons = ap.add_argument_group("Consistency measurement")
    g_cons.add_argument("--consistency-mode", choices=["cp", "reorg", "both"], default="both", help="Common prefix only (cp), deep reorg only (reorg), or both.")
    g_cons.add_argument("--consistency-stride", type=int, default=0, help="Stride (rounds) between consistency checks (0 = every round).")

    # Output / execution
    g_out = ap.add_argument_group("Output/Execution")
    g_out.add_argument("--outfile", type=str, default="consistency_grid.csv", help="CSV output path.")
    g_out.add_argument("--append", action="store_true", help="Append to CSV if it already exists.")
    g_out.add_argument("--cores", type=int, default=1, help="Number of worker processes (1 = serial).")

    args = ap.parse_args()

    if args.help_args:
        _print_args_recap(ap)
        sys.exit(0)

    return args

# =========================
# Build c list
# =========================
def make_c_list(args) -> List[float]:
    """
    Return the list of c values:
      - if --c-values is provided, parse that;
      - else if --c-min/max/steps are provided, build a linear range;
      - else return a small demo list.
    """
    if args.c_values:
        vals = []
        for tok in args.c_values.split(","):
            tok = tok.strip()
            if tok:
                vals.append(float(tok))
        return vals
    if args.c_min is not None and args.c_max is not None and args.c_steps:
        if args.c_steps <= 1:
            return [args.c_min]
        step = (args.c_max - args.c_min) / (args.c_steps - 1)
        return [round(args.c_min + i * step, 6) for i in range(args.c_steps)]
    # default demo (good for a quick smoke test)
    return [0.5, 0.7, 1.0, 1.5, 2, 3]

# =========================
# Pretty duration
# =========================
def _fmt_duration(seconds: float) -> str:
    """
    Convert seconds to a short string like 'Xm Ys' or 'Hh Mm Ss'.
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}h {m}m {s:.1f}s" if h > 0 else f"{m}m {s:.1f}s"

# =========================
# Main orchestration
# =========================
def main():
    t0 = time.perf_counter()  # ---- START TIMER ----

    args = parse_args()
    c_list = make_c_list(args)

    # Prepare CSV (append or overwrite) with lazy writer initialization
    need_header = not (args.append and os.path.exists(args.outfile))
    out = open(args.outfile, "a" if args.append else "w", newline="")
    writer = None

    # Build tasks (c, rho) according to the selected grid strategy
    tasks = []
    for c in c_list:
        rho_values, rho_old, rho_new = build_rho_grid(
            c=c,
            sigma=args.sigma,
            n_points=args.rho_points,
            eps=args.rho_eps,
            split=args.rho_split,
            rho_max=args.rho_max,
            rho_step=args.rho_step,
            rho_density=args.rho_density,
            rho_start=args.rho_start,
        )
        # Compact telemetry: grid summary for this c
        print(
            f"[c={c:.3f}] rho_old*={rho_old}  rho_new*={rho_new}  "
            f"#rho={len(rho_values)} (start={args.rho_start})"
        )
        for rho in rho_values:
            tasks.append((c, rho))

    # Execute: parallel (ProcessPool) or serial
    if args.cores > 1:
        with ProcessPoolExecutor(max_workers=args.cores) as ex:
            futs = {ex.submit(eval_point, c, rho, args): (c, rho) for (c, rho) in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                if r is None:
                    continue
                if writer is None:
                    # initialize header on the first written row
                    writer = csv.DictWriter(out, fieldnames=list(r.keys()))
                    if need_header:
                        writer.writeheader()
                        need_header = False
                writer.writerow(r)
                out.flush()
                # Progress: index, c, rho_eff, T used, successes/reps, p̂
                print(
                    f"  -> ({i}/{len(tasks)}) c={r['c']:.3f}  rho_eff={r['rho_effective']:.2f} "
                    f"T={r['max_round']} success={r['successes']}/{r['reps']}  p̂={r['p_hat']:.4f}"
                )
    else:
        # Serial execution with streaming CSV writes
        for i, (c, rho) in enumerate(tasks, 1):
            r = eval_point(c, rho, args)
            if r is None:
                continue
            if writer is None:
                writer = csv.DictWriter(out, fieldnames=list(r.keys()))
                if need_header:
                    writer.writeheader()
                    need_header = False
            writer.writerow(r)
            out.flush()
            print(
                f"  -> ({i}/{len(tasks)}) c={r['c']:.3f}  rho_eff={r['rho_effective']:.4f}  "
                f"success={r['successes']}/{r['reps']}  p̂={r['p_hat']:.3f}"
            )

    out.close()
    print(f"✔ Saved CSV to: {args.outfile}")

    dt = time.perf_counter() - t0            # ---- STOP TIMER ----
    print(f"⏱ Total time: {_fmt_duration(dt)}")

if __name__ == "__main__":
    main()
