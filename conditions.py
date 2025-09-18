import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from matplotlib.widgets import Slider
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D


# --- Fixed-point functions ---
def f1(x, c, s):
    """Pass & Shi fixed-point equation."""
    return (1 - (2 * (1 - s - x)) / c) * (1 - s - x) - x

def f2(x, c, s):
    """Our condition fixed-point equation."""
    return (1 - s - x) * np.exp(-(2 * (1 - s - x)) / c) - x

def safe_brentq(fun, a, b, args=(), scan_points=2001):
    """
    Robust root finder on [a, b]:
    - Tries brentq directly if the endpoints have opposite signs.
    - Otherwise scans the interval on a uniform grid to locate a sign change
      and then calls brentq on the first bracketing sub-interval.

    Returns
    -------
    float
        A root in [a, b].
    Raises
    ------
    ValueError
        If no sign change can be found on [a, b].
    """
    fa, fb = fun(a, *args), fun(b, *args)

    if np.isfinite(fa) and fa == 0:
        return a
    if np.isfinite(fb) and fb == 0:
        return b

    if np.isfinite(fa) and np.isfinite(fb) and np.sign(fa) * np.sign(fb) < 0:
        return brentq(fun, a, b, args=args)

    xs = np.linspace(a, b, scan_points)
    vals = np.array([fun(x, *args) for x in xs], dtype=float)
    vals[~np.isfinite(vals)] = np.nan

    sgn = np.sign(vals)
    ok = ~np.isnan(vals)
    idxs = np.where(ok[:-1] & ok[1:] & (sgn[:-1] * sgn[1:] < 0))[0]
    if idxs.size == 0:
        raise ValueError("no sign change found")

    i = idxs[0]
    aa, bb = xs[i], xs[i + 1]
    return brentq(fun, aa, bb, args=args)


# --- c-grids (shared by f1 and f2; include c < 2) ---
rng = np.random.default_rng(0)
c_values_between_0_and_2 = np.linspace(0.1, 1.9, 10)
c_values_between_2_and_3 = np.linspace(2, 3, 7)
c_values_between_3_and_4 = np.linspace(3.2, 4, 4)
c_values_above_4 = rng.exponential(scale=20, size=10) + 4
c_values_specific = np.array([1.2, 1.5, 10, 20, 60, 100])

c_values_all = np.unique(
    np.concatenate(
        [
            c_values_between_0_and_2,
            c_values_between_2_and_3,
            c_values_between_3_and_4,
            c_values_above_4,
            c_values_specific,
        ]
    )
)
c_values_all.sort()

# Dense log-spaced grid for smooth lines
c_min, c_max = c_values_all.min(), c_values_all.max()
c_dense = np.geomspace(c_min, c_max, 800)

# --- Domain for rho (root variable) ---
eps = 1e-12
lower_bound = 0.0 + eps
upper_bound = 1.0 - eps


def x_formatter(x, pos):
    """Tick formatter for c-axis: integers >= 1, compact decimals otherwise."""
    return "{:.0f}".format(x) if x >= 1 else ("{:.1f}".format(x).rstrip("0").rstrip("."))


def compute_roots_over_c(fun, c_array, s, require_cond=None):
    """
    Compute roots across c_array at fixed sigma s.
    If `require_cond(c)` is provided and False, returns NaN for that c.

    Parameters
    ----------
    fun : callable
        Function of the form fun(x, c, s).
    c_array : array-like
        Values of c to evaluate.
    s : float
        Sigma value.
    require_cond : callable or None
        Optional predicate on c; when False, skip and return NaN.

    Returns
    -------
    np.ndarray
        Roots aligned with c_array (NaN where bracketing fails or condition not met).
    """
    roots = []
    for c in c_array:
        if (require_cond is not None) and (not require_cond(c)):
            roots.append(np.nan)
            continue
        try:
            r = safe_brentq(fun, lower_bound, upper_bound, args=(c, s))
            roots.append(r)
        except ValueError:
            roots.append(np.nan)
    return np.array(roots, dtype=float)


def make_figure():
    """Create figure, slider, and wire up the update callback."""
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.35)

    ax_slider_s = plt.axes([0.25, 0.05, 0.5, 0.03])
    # valinit must lie strictly within (0, 1) to avoid numerical issues at the endpoints
    slider_s = Slider(ax_slider_s, r"$\sigma$", 1e-6, 1 - 1e-6, valinit=0.0, valstep=0.01)

    def update(_):
        s = float(slider_s.val)

        # f1: analytical existence condition c > 2(1 - s)
        threshold = 2.0 * (1.0 - s)
        cond_f1 = lambda c: (c > threshold)

        # Roots at the original c grid (markers)
        x_roots_f1_pts = compute_roots_over_c(f1, c_values_all, s, require_cond=cond_f1)
        x_roots_f2_pts = compute_roots_over_c(f2, c_values_all, s, require_cond=None)

        # Roots over a dense c grid (smooth lines)
        x_roots_f1_dense = compute_roots_over_c(f1, c_dense, s, require_cond=cond_f1)
        x_roots_f2_dense = compute_roots_over_c(f2, c_dense, s, require_cond=None)

        ax.clear()

        # Smooth lines (legend shows only lines)
        line_f1, = ax.plot(c_dense, x_roots_f1_dense, linestyle="-", linewidth=0.7, color="b", alpha=0.8)
        line_f2, = ax.plot(c_dense, x_roots_f2_dense, linestyle="-", linewidth=0.7, color="r", alpha=0.8)

        # Markers at the original points (not included in legend)
        ax.plot(c_values_all, x_roots_f1_pts, marker="o", linestyle="None", markersize=2, color="b", label="_nolegend_")
        ax.plot(c_values_all, x_roots_f2_pts, marker="o", linestyle="None", markersize=2, color="r", label="_nolegend_")

        # Axes and styling
        ax.set_xscale("log")
        ax.set_xlabel("c")
        ax.set_ylabel(r"$\rho$")
        ax.set_title("Maximum value of " + r"$\rho$" + " for which consistency still holds")
        ax.grid(True, which="both", linestyle="--", linewidth=0.4)

        ax.set_xticks([0.5, 1, 2, 4, 10, 30, 60, 100])
        ax.xaxis.set_major_formatter(FuncFormatter(x_formatter))
        ax.set_xlim(c_min * 0.9, c_max * 1.1)

        # Dynamic y-limits (based on dense data)
        all_dense = np.concatenate([x_roots_f1_dense, x_roots_f2_dense])
        finite_dense = all_dense[np.isfinite(all_dense)]
        if finite_dense.size > 0:
            ymin = float(finite_dense.min())
            ymax = float(finite_dense.max())
            delta = (ymax - ymin) * 0.05 if ymax > ymin else 0.05
            ax.set_ylim(max(0.0, ymin - delta), min(1.0, ymax + delta))
        else:
            ax.set_ylim(0.0, 1.0)

        # Legend: lines only (no markers)
        legend_elements = [
            Line2D([0], [0], color="b", lw=1.3, label="Pass and Shi"),
            Line2D([0], [0], color="r", lw=1.3, label="Ours"),
        ]
        ax.legend(handles=legend_elements, loc="best")

        plt.draw()

    # Initialize and connect
    update(None)
    slider_s.on_changed(update)
    fig._slider_s = slider_s
    fig._update_cb = update

    # Equations (annotation)
    plt.figtext(
        0.5,
        0.18,
        r"$f1(\rho,c,\sigma)=\left(1-\frac{2(1-\sigma-\rho)}{c}\right)(1-\sigma-\rho)-\rho$",
        ha="center",
        color="black",
        fontsize=10,
    )
    plt.figtext(
        0.5,
        0.12,
        r"$f2(\rho,c,\sigma)=(1-\sigma-\rho)\,e^{-\frac{2(1-\sigma-\rho)}{c}}-\rho$",
        ha="center",
        color="black",
        fontsize=10,
    )
    return fig, slider_s

def main():
    fig, slider_s = make_figure()
    plt.show()

if __name__ == "__main__":
    main()
