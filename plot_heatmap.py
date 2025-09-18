# plot_consistency_heatmap.py
"""
Visualize the probability of k-inconsistency (Balancing Attack) as a heatmap
over (c, rho) from a CSV produced by the sweep script. Optionally overlays the
theoretical thresholds ("Pass and Shi" vs "Ours") for a given sigma.

Expected CSV columns (at least):
  - c
  - rho_effective (or rho)
  - p_hat (or phat)
Optional columns:
  - sigma
  - k

Usage examples:
  python plot_consistency_heatmap.py --infile consistency_grid.csv
  python plot_consistency_heatmap.py --infile grid.csv --sigma 0.1 --out heat.png
  python plot_consistency_heatmap.py --infile grid.csv --help-recap
"""

import argparse
import math
import csv
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import brentq
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter, LogLocator

# =========================
# INTERNAL HEATMAP CONFIG
# =========================
# Color scaling mode: "log" | "power" | "quantile" | "linear"
COLOR_SCALE = "quantile"
# Colormap name
CMAP_NAME   = "inferno"
# Scale parameters
GAMMA       = 0.35   # used if COLOR_SCALE == "power"
QMAX        = 0.99   # top quantile if COLOR_SCALE == "quantile"
LOG_EPS     = 1e-4   # to avoid log(0); exact zeros map to the 'under' color


# ------------------------------
# Roots and theoretical thresholds
# ------------------------------
def _root_with_scan(f, lo: float, hi: float, steps: int = 400) -> Optional[float]:
    """Find a sign change by scanning, then refine with Brent's method."""
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


def rho_star_new(c: float, sigma: float) -> Optional[float]:
    """New condition: h * exp(-2h/c) - rho = 0, with h = 1 - sigma - rho."""
    if c <= 0 or sigma >= 1:
        return None
    eps = 1e-9
    lo, hi = eps, max(eps, 1 - sigma - eps)

    def f_new(rho: float) -> float:
        h = 1 - sigma - rho
        if h <= 0:
            return -1.0
        return h * math.exp(-2 * h / c) - rho

    return _root_with_scan(f_new, lo, hi)


def rho_star_old(c: float, sigma: float) -> Optional[float]:
    """Previous (Pass and Shi): (1 - 2h/c) * h - rho = 0, with h = 1 - sigma - rho."""
    if c <= 0 or sigma >= 1:
        return None
    eps = 1e-9
    lo, hi = eps, max(eps, 1 - sigma - eps)

    def f_old(rho: float) -> float:
        h = 1 - sigma - rho
        if h <= 0:
            return -1.0
        return (1 - 2 * h / c) * h - rho

    return _root_with_scan(f_old, lo, hi)


# ------------------------------
# CSV loading
# ------------------------------
def load_grid(infile: str):
    """
    Load grid data from CSV (output of sweep_consistency.py).
    Columns:
      - c
      - rho_effective (or rho)
      - p_hat (or phat)
    Optional:
      - sigma (first valid value is used if --sigma is not provided)
      - k     (first valid value is read for the plot title)
    """
    C, RHO, P, SIGMA = [], [], [], []
    k_value: Optional[str] = None

    with open(infile, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                c = float(row.get("c", row.get("C", "nan")))
                rho = float(row.get("rho_effective", row.get("rho", "nan")))
                phat = float(row.get("p_hat", row.get("phat", "nan")))
            except Exception:
                continue
            if not (np.isfinite(c) and np.isfinite(rho) and np.isfinite(phat)):
                continue

            C.append(c)
            RHO.append(rho)
            P.append(phat)

            s = row.get("sigma", None)
            if s is not None:
                try:
                    SIGMA.append(float(s))
                except Exception:
                    pass

            if k_value is None:
                k_candidate = row.get("k", None)
                if k_candidate is not None:
                    k_value = str(k_candidate)

    sigma = SIGMA[0] if SIGMA else None
    return np.array(C), np.array(RHO), np.array(P), sigma, k_value


# ------------------------------
# Gridding → pcolormesh
# ------------------------------
def centers_to_edges_linear(vals: np.ndarray) -> np.ndarray:
    """Given sorted center positions, compute linear bin edges."""
    vals = np.sort(np.unique(vals))
    if len(vals) == 1:
        v = vals[0]
        return np.array([v - 0.5, v + 0.5])
    mids = (vals[:-1] + vals[1:]) / 2.0
    first_edge = vals[0] - (mids[0] - vals[0])
    last_edge = vals[-1] + (vals[-1] - mids[-1])
    return np.concatenate([[first_edge], mids, [last_edge]])


def centers_to_edges_geometric(vals: np.ndarray) -> np.ndarray:
    """For log-scaled x-axis: use geometric means for edges."""
    vals = np.sort(np.unique(vals))
    if len(vals) == 1:
        v = vals[0]
        return np.array([v / math.sqrt(10), v * math.sqrt(10)])
    mids = np.sqrt(vals[:-1] * vals[1:])
    first_edge = vals[0] ** 2 / mids[0]
    last_edge = vals[-1] ** 2 / mids[-1]
    return np.concatenate([[first_edge], mids, [last_edge]])


def build_matrix(C: np.ndarray, R: np.ndarray, P: np.ndarray):
    """
    Average p-hat values for duplicate (c, rho) cells and return:
      - C_edges (log-scale bins),
      - R_edges (linear bins),
      - Z matrix (mean p-hat per cell; NaN for empty).
    """
    c_vals = np.sort(np.unique(C))
    r_vals = np.sort(np.unique(R))
    c_map = {v: i for i, v in enumerate(c_vals)}
    r_map = {v: i for i, v in enumerate(r_vals)}
    Z = np.full((len(r_vals), len(c_vals)), np.nan)
    COUNT = np.zeros_like(Z, dtype=int)

    for c, r, p in zip(C, R, P):
        i = r_map[r]
        j = c_map[c]
        if np.isnan(Z[i, j]):
            Z[i, j] = 0.0
        Z[i, j] += p
        COUNT[i, j] += 1

    mask = COUNT > 0
    Z[mask] = Z[mask] / COUNT[mask]

    C_edges = centers_to_edges_geometric(c_vals)  # X (log)
    R_edges = centers_to_edges_linear(r_vals)     # Y (linear)
    return C_edges, R_edges, Z


# ------------------------------
# Plot
# ------------------------------
def plot_heatmap(infile: str, sigma_arg: Optional[float], outpng: Optional[str] = None):
    """Main plotting routine: load CSV, build grid, render heatmap + theory curves."""
    C, RHO, P, sigma_in, k = load_grid(infile)
    if C.size == 0:
        raise RuntimeError("No valid data found in CSV.")

    # sigma used for overlay curves (from CLI, else from CSV, else fallback)
    sigma = sigma_arg if sigma_arg is not None else sigma_in
    if sigma is None:
        sigma = 0.10

    C_edges, R_edges, Z = build_matrix(C, RHO, P)

    fig, ax = plt.subplots(figsize=(8, 5))

    # --- Normalization / colormap according to COLOR_SCALE ---
    cmap = plt.get_cmap(CMAP_NAME).copy()
    norm = None
    Z_plot = Z

    if COLOR_SCALE.lower() == "log":
        # push non-positive to LOG_EPS; keep NaNs as NaNs (unsampled cells)
        Z_plot = Z.copy()
        zeros_mask = (np.isfinite(Z_plot)) & (Z_plot <= 0)
        Z_plot[zeros_mask] = LOG_EPS
        vmax = np.nanmax(Z_plot) if np.isfinite(Z_plot).any() else 1.0
        vmin = LOG_EPS * 2.0
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        cmap.set_under("whitesmoke")

    elif COLOR_SCALE.lower() == "power":
        vmax = np.nanmax(Z) if np.isfinite(Z).any() else 1.0
        norm = mcolors.PowerNorm(gamma=GAMMA, vmin=0.0, vmax=vmax)

    elif COLOR_SCALE.lower() == "quantile":
        Zpos = Z[np.isfinite(Z) & (Z > 0)]
        if Zpos.size >= 2:
            levels = np.quantile(Zpos, np.linspace(0.0, QMAX, 12))
            levels = np.unique(np.r_[0.0, levels])  # ensure 0 is present
            norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=256, extend="max")
        else:
            norm = None  # fallback to linear

    # Heatmap
    quad = ax.pcolormesh(
        C_edges,
        R_edges,
        Z_plot,
        shading="auto",
        cmap=cmap,
        norm=norm,
        vmin=None if norm else 0.0,
        vmax=None if norm else 1.0,
    )
    cbar = plt.colorbar(quad, ax=ax, pad=0.01)
    cbar.set_label("Pr[Balancing Attack success]" + (" [log]" if COLOR_SCALE.lower() == "log" else ""))

    # Axes
    ax.set_xscale("log")
    ax.set_xlabel("c")
    ax.set_ylabel(r"$\rho$")

    # Pleasant log ticks for c
    xticks_pref = np.array([0.5, 1, 2, 3, 4, 10, 20, 60, 100])
    xmin, xmax = C_edges.min(), C_edges.max()
    xt = [x for x in xticks_pref if xmin <= x <= xmax]
    if xt:
        ax.xaxis.set_major_locator(FixedLocator(xt))
        ax.xaxis.set_major_formatter(FixedFormatter([str(int(x)) if x >= 1 else f"{x:.1f}" for x in xt]))
        # Hide minor labels like "6×10⁻¹"
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.xaxis.set_minor_formatter(NullFormatter())

    # Overlay theoretical thresholds
    c_dense = np.logspace(np.log10(C_edges.min() * 1.001), np.log10(C_edges.max() * 0.999), 400)
    rho_new = np.array([rho_star_new(cval, sigma) for cval in c_dense], dtype=float)
    rho_old = np.array([rho_star_old(cval, sigma) for cval in c_dense], dtype=float)

    def plot_masked(x, y, **kw):
        """Plot only the finite segments of a piecewise-defined curve."""
        y = np.array(y, dtype=float)
        mask = np.isfinite(y)
        if not np.any(mask):
            return
        starts = np.where(np.diff(mask.astype(int)) == 1)[0] + 1
        stops = np.where(np.diff(mask.astype(int)) == -1)[0] + 1
        if mask[0]:
            starts = np.r_[0, starts]
        if mask[-1]:
            stops = np.r_[stops, len(mask)]
        for s, t in zip(starts, stops):
            ax.plot(x[s:t], y[s:t], **kw)

    plot_masked(c_dense, rho_new, color="#2bb41f", lw=1.6, label="Ours")
    plot_masked(c_dense, rho_old, color="#1f77b4", lw=1.6, ls="--", label="Pass and Shi")

    ax.set_ylim(max(0.0, R_edges.min()), min(1.0 - sigma, R_edges.max()))
    ax.grid(True, which="both", ls="--", lw=0.1, alpha=0.2)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(fr"$\hat p$ of $k$-Inconsistency $(\sigma={sigma:g},\ k={k if k is not None else '? '})$")

    if outpng:
        plt.savefig(outpng, dpi=180, bbox_inches="tight")
        print(f"Saved: {outpng}")
    plt.show()


# ------------------------------
# CLI
# ------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    """
    Create and return the CLI parser (used both for --help and --help-recap).
    """
    ap = argparse.ArgumentParser(
        description="Heatmap of inconsistency probability (Balancing Attack) over (c, rho).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--infile",
        required=True,
        help="CSV produced by sweep_consistency (grid). Must contain c, rho_effective (or rho), p_hat (or phat).",
    )
    ap.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Sigma to use for theoretical curves. If omitted, use the first value found in CSV; "
             "fallback to 0.10 if CSV has no sigma column.",
    )
    ap.add_argument(
        "--out",
        dest="out",
        default=None,
        help="Optional PNG output path. If not provided, the plot is only shown on screen.",
    )
    ap.add_argument(
        "--help-recap",
        action="store_true",
        help="Print a compact recap of all supported arguments (names, defaults, choices) and exit.",
    )
    return ap


def print_args_recap(parser: argparse.ArgumentParser) -> None:
    """
    Print a compact, human-friendly recap of all CLI arguments supported by this script.
    """
    print("\nArguments recap:")
    for action in parser._actions:
        if not action.option_strings:
            # positional or help; skip positionals (none here) and the built-in help
            continue
        names = ", ".join(action.option_strings)
        default = None if action.default is argparse.SUPPRESS else action.default
        choices = getattr(action, "choices", None)
        help_text = (action.help or "").strip()
        extra = []
        if choices:
            extra.append(f"choices={list(choices)}")
        if default not in (None, False):
            extra.append(f"default={default}")
        extra_str = f" ({'; '.join(extra)})" if extra else ""
        print(f"  {names}: {help_text}{extra_str}")
    print()


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.help_recap:
        print_args_recap(parser)
        return

    plot_heatmap(args.infile, args.sigma, args.out)


if __name__ == "__main__":
    main()
