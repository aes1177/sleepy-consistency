import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

# =========================
# Global figure settings
# =========================
FIGSIZE = (9, 6)
DPI = 150

mpl.rcParams["figure.figsize"] = FIGSIZE      # default size (inches)
mpl.rcParams["figure.dpi"] = DPI              # on-screen dpi
mpl.rcParams["savefig.dpi"] = DPI             # saved dpi

# =========================
# Global state for recurrences (initialized)
# =========================
honest_prob = 0.0
adversary_prob = 0.0
delta_silent_prob = 0.0
h_delta_prob = 0.0
a_delta_prob = 0.0
union_prob = 0.0

# =========================
# Memoization helpers
# =========================
def clear_memo():
    """Clear memoization caches for P and Delta recurrences."""
    global memo_P, memo_Delta
    memo_P, memo_Delta = {}, {}

memo_P, memo_Delta = {}, {}

# =========================
# Recurrences
# =========================
def calculate_Delta(k, s):
    """
    Recurrence for Delta(k, s).

    Notes
    -----
    This function relies on the following globals (set in the driver loops):
      - union_prob
      - h_delta_prob
      - delta_silent_prob
      - calculate_P (mutual recursion)
    """
    key = (k, s)
    if key in memo_Delta:
        return memo_Delta[key]

    if s - 1 >= k:
        res = 1.0
    elif s == -1:
        res = 0.0
    else:
        res = (
            union_prob * calculate_P(k - 1, s)
            + h_delta_prob * calculate_Delta(k - 1, s - 1)
            + delta_silent_prob * calculate_P(k - 1, s - 1)
        )

    memo_Delta[key] = res
    return res

def calculate_P(k, s):
    """
    Recurrence for P(k, s).

    Notes
    -----
    This function relies on the following globals (set in the driver loops):
      - honest_prob
      - adversary_prob
      - calculate_Delta (mutual recursion)
    """
    key = (k, s)
    if key in memo_P:
        return memo_P[key]

    if s >= k:
        res = 1.0
    elif s == -1:
        res = 0.0
    else:
        delta_state = calculate_Delta(k, s)
        res = honest_prob * delta_state + adversary_prob * calculate_P(k, s + 1)

    memo_P[key] = res
    return res

# =========================
# Parameters and initial setup
# =========================
c_values = [1, 4, 60]
sleepy_fraction = 0.1
adversary_fractions_for_plot = [0.25, 0.40]
k_values = np.arange(2, 16, 2)

def plot_P0k():
    """Plot P0(k) for different c and adversary fractions."""
    global honest_prob, adversary_prob, delta_silent_prob, h_delta_prob, a_delta_prob, union_prob

    # color mapping for c
    color_map = {1: "purple", 4: "orange", 60: "green"}

    plt.figure()  # uses FIGSIZE/DPI defaults

    for adversary_fraction in adversary_fractions_for_plot:
        if adversary_fraction == 0.25:
            line_style = "--"
            adv_label = "25% Adv"
        else:
            line_style = "-"
            adv_label = "40% Adv"

        for c in c_values:
            # set global probabilities used by the recurrences
            adversary_prob = adversary_fraction
            honest_prob = 1 - sleepy_fraction - adversary_prob

            # transitions / event probabilities
            delta_silent_prob = math.exp(-1 / c)
            h_delta_prob = 1 - math.exp(-(honest_prob / 2) / c)
            a_delta_prob = 1 - math.exp(-adversary_prob / c)
            union_prob = 1 - math.exp(-(adversary_prob + (honest_prob / 2)) / c)

            clear_memo()
            results = [calculate_P(k, 0) for k in k_values]

            plt.plot(
                k_values,
                results,
                line_style,
                label=f"{adv_label}" if c == c_values[0] else "",
                color=color_map[c],
            )

    # legend
    custom_lines = [
        Line2D([0], [0], color="black", lw=2, linestyle="-", label="40% Adv"),
        Line2D([0], [0], color="black", lw=2, linestyle="--", label="25% Adv"),
        Line2D([0], [0], color="purple", lw=2, linestyle="-", label="c=1"),
        Line2D([0], [0], color="orange", lw=2, linestyle="-", label="c=4"),
        Line2D([0], [0], color="green", lw=2, linestyle="-", label="c=60"),
    ]
    plt.legend(handles=custom_lines, loc="best")
    plt.xlabel(r"$k$")
    plt.ylabel(r"$P_0(k)$")
    plt.yscale("log")
    plt.title("Probability of winning the Balancing Attack (k)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =========================
# T-consistency section
# =========================
def compute_T_consistency():
    """
    Compute T (smallest k such that P(k, 0) < P_max) for selected (a, c) pairs,
    then do a dense sweep over c>1 and save CSV + plot.
    """
    global honest_prob, adversary_prob, delta_silent_prob, h_delta_prob, a_delta_prob, union_prob

    P_max = 1e-6  # acceptable threshold
    T_values = {}
    c_values_final = [2, 4, 10, 60]      # c>1
    adversary_fractions_final = [0.10, 0.25, 0.40]
    k_cap = 1000                         # guardrail to avoid infinite loops

    # Explicit points
    for adversary_fraction in adversary_fractions_final:
        for c in c_values_final:
            # set globals for the recurrences
            adversary_prob = adversary_fraction
            honest_prob = 1 - sleepy_fraction - adversary_prob
            delta_silent_prob = math.exp(-1 / c)
            h_delta_prob = 1 - math.exp(-(honest_prob / 2) / c)
            a_delta_prob = 1 - math.exp(-adversary_prob / c)
            union_prob = 1 - math.exp(-(adversary_prob + (honest_prob / 2)) / c)

            clear_memo()
            T_found = False
            k = 2
            while not T_found and k <= k_cap:
                result = calculate_P(k, 0)
                if result < P_max:
                    T_values[(adversary_fraction, c)] = k
                    T_found = True
                k += 1
            if not T_found:
                T_values[(adversary_fraction, c)] = float("nan")

    print("Calculated T-consistency (smallest k with P(k, 0) < P_max):")
    for (a_frac, c), T in T_values.items():
        print(f"For adversary = {a_frac}, c = {c}: T = {T}")

    # Dense sweep on c>1 (log x), save CSV, and plot
    c_dense = np.logspace(np.log10(1.0), np.log10(60.0), 60)
    c_points = np.unique(np.concatenate([c_dense, np.array(c_values_final, dtype=float)]))
    c_points = c_points[c_points >= 1.0]
    c_points.sort()

    rows = []
    series_by_a = {}

    for adversary_fraction in adversary_fractions_final:
        c_list, T_list = [], []
        for c in c_points:
            # set globals
            adversary_prob = adversary_fraction
            honest_prob = 1 - sleepy_fraction - adversary_prob
            delta_silent_prob = math.exp(-1 / c)
            h_delta_prob = 1 - math.exp(-(honest_prob / 2) / c)
            a_delta_prob = 1 - math.exp(-adversary_prob / c)
            union_prob = 1 - math.exp(-(adversary_prob + (honest_prob / 2)) / c)

            clear_memo()
            T_found = False
            k = 2
            while not T_found and k <= k_cap:
                result = calculate_P(k, 0)
                if result < P_max:
                    T = k
                    T_found = True
                else:
                    k += 1
            if not T_found:
                T = float("nan")

            c_list.append(c)
            T_list.append(T)
            rows.append(
                {
                    "sleepy_fraction": sleepy_fraction,
                    "adversary_fraction": adversary_fraction,
                    "c": c,
                    "T": T,
                    "P_max": P_max,
                }
            )
        series_by_a[adversary_fraction] = (np.array(c_list), np.array(T_list))

    # Save CSV
    csv_path = "T_consistency_grid.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sleepy_fraction", "adversary_fraction", "c", "T", "P_max"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV: {csv_path}")

    # Plot: T vs c (log x) per adversary fraction
    plt.figure()  # uses FIGSIZE/DPI defaults
    colors = {0.10: "#1f77b4", 0.25: "#ff7f0e", 0.40: "#2ca02c"}
    markers = {0.10: "o", 0.25: "s", 0.40: "D"}

    for adversary_fraction in adversary_fractions_final:
        xs, ys = series_by_a[adversary_fraction]
        valid = np.isfinite(ys)
        xs, ys = xs[valid], ys[valid]
        plt.plot(
            xs,
            ys,
            "-",
            lw=1,
            color=colors[adversary_fraction],
            label=f"a = {int(adversary_fraction * 100)}%",
        )
        plt.scatter(xs, ys, s=12, color=colors[adversary_fraction], marker=markers[adversary_fraction])

    plt.xscale("log")
    ax = plt.gca()
    ticks = [1, 2, 4, 10, 30, 60]
    ax.set_xlim(1, 60)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    plt.xlabel("c")
    plt.ylabel("T")
    plt.title("T-consistency threshold vs c")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(title="Adversary fraction")
    plt.tight_layout()
    plt.show()

def main():
    plot_P0k()
    compute_T_consistency()

if __name__ == "__main__":
    main()
