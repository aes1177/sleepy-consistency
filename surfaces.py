import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# ===========================
#  EDGE HANDLING TOGGLE
# ===========================
# If True, ignore values too close to the theoretical boundary c = 2(1 - sigma)
# by shifting the boundary outward by a relative margin. If False, plot the full domain.
USE_SMOOTH_EDGE = False
EDGE_MARGIN_REL = 0.10  # relative margin (e.g., 0.03, 0.10, 0.15). Ignored if USE_SMOOTH_EDGE=False.

# --- Base fixed-point functions ---
def f1(x, c, s):
    """Fixed-point equation from Pass & Shi."""
    return (1 - (2 * (1 - s - x)) / c) * (1 - s - x) - x

def f2(x, c, s):
    """Our fixed-point equation (exponential term)."""
    return (1 - s - x) * np.exp(-(2 * (1 - s - x)) / c) - x


# --- Closed-form stable solution for f1 ---
# For f1, y = (c/2) * (1 - sqrt(1 - 2(1 - s)/c)), then rho = 1 - s - y.
def rho_f1_closed_masked(c, s, margin_rel=0.0):
    """
    Return rho(c, s) for Pass & Shi with an optional mask that ignores points
    too close to the theoretical boundary c = 2(1 - s).

    Parameters
    ----------
    c, s : array-like
        Broadcastable arrays of the parameter c and sigma (s).
    margin_rel : float, default 0.0
        Relative outward shift of the boundary. If 0.0, use the exact domain;
        if > 0, ignore (mask out) points with c < (1 + margin_rel) * 2(1 - s).

    Returns
    -------
    rho_masked : ndarray
        Array of rho values with np.nan where the point is masked or invalid.
    c_edge_shift : ndarray
        The shifted boundary curve (same shape as inputs) for reference.
    """
    C, S = np.broadcast_arrays(np.asarray(c, float), np.asarray(s, float))
    rho = np.full_like(C, np.nan, dtype=float)

    # Theoretical boundary and shifted boundary (if margin_rel > 0)
    c_edge = 2.0 * (1.0 - S)
    c_edge_shift = (1.0 + margin_rel) * c_edge

    # Compute only where sufficiently far from the (shifted) boundary
    mask = C >= c_edge_shift
    if not np.any(mask):
        return rho, c_edge_shift  # all NaN; still return the shifted boundary

    sqrt_arg = 1.0 - 2.0 * (1.0 - S[mask]) / C[mask]
    sqrt_arg = np.clip(sqrt_arg, 0.0, None)

    y = 0.5 * C[mask] * (1.0 - np.sqrt(sqrt_arg))
    rho_val = 1.0 - S[mask] - y

    ok = (rho_val >= 0.0) & (rho_val <= 1.0)
    rho_masked = np.full_like(C, np.nan, dtype=float)
    rho_masked[mask] = np.where(ok, rho_val, np.nan)
    return rho_masked, c_edge_shift


# --- c-grid for f2 (heterogeneous coverage across scales) ---
rng = np.random.default_rng(0)
c_values_between_0_and_2 = np.linspace(0.1, 1.9, 10)
c_values_between_2_and_3 = np.linspace(2, 3, 7)
c_values_between_3_and_4 = np.linspace(3.2, 4, 4)
c_values_above_4 = rng.exponential(scale=20, size=10) + 4
c_values_specific = np.array([10, 20, 60, 100])

c_values_all = np.unique(np.concatenate([
    c_values_between_0_and_2,
    c_values_between_2_and_3,
    c_values_between_3_and_4,
    c_values_above_4,
    c_values_specific
]))
c_values_all.sort()

# --- f1: dense regular grid in log(c) and sigma ---
cmin, cmax = c_values_all.min(), c_values_all.max()
c_values_f1_dense = np.geomspace(cmin, cmax, 400)
sigma_values_f1_dense = np.linspace(0.0, 1.0, 201)

# --- f2: moderate grid in sigma (paired with heterogeneous c_values_all) ---
sigma_values_f2 = np.linspace(0.0, 1.0, 201)

# --- brentq bounds (avoid endpoints) ---
eps = 1e-12
lower_bound = 0.0 + eps
upper_bound = 1.0 - eps

def main():
    # --- Compute surfaces ---
    # f1 via closed form (optionally "smoothed" boundary)
    delta_rel = EDGE_MARGIN_REL if USE_SMOOTH_EDGE else 0.0
    C1, S1 = np.meshgrid(c_values_f1_dense, sigma_values_f1_dense)
    rho_values_f1, C1_edge_shift = rho_f1_closed_masked(C1, S1, margin_rel=delta_rel)

    # f2 via numerical root-finding over (c_values_all × sigma_values_f2)
    rho_values_f2 = np.full((len(sigma_values_f2), len(c_values_all)), np.nan)
    for i, c in enumerate(c_values_all):
        for j, s in enumerate(sigma_values_f2):
            try:
                rho_values_f2[j, i] = brentq(f2, lower_bound, upper_bound, args=(c, s))
            except ValueError:
                rho_values_f2[j, i] = np.nan

    # --- transform to log10(c) for plotting ---
    C1_log = np.log10(C1)
    C2_log, S2 = np.meshgrid(np.log10(c_values_all), sigma_values_f2)

    # --- figure with two 3D surfaces ---
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(14, 7))

    title_suffix = " (smoothed edge)" if USE_SMOOTH_EDGE else " (full domain)"

    # f1 surface
    surf1 = ax1.plot_surface(C1_log, S1, rho_values_f1, cmap=cm.viridis, linewidth=0, antialiased=True)
    ax1.set_title('Pass and Shi' + title_suffix + ': ' + r'$f1(\rho,\sigma,c)=\left(1-\frac{2(1-\sigma-\rho)}{c}\right)(1-\sigma-\rho)-\rho$', color='b')
    ax1.set_xlabel('c (log scale)', color='#00008B')
    ax1.set_ylabel(r'$\sigma$', color='#00008B')
    ax1.set_zlabel('Maximum value of ' + r'$\rho$', color='#00008B')
    ax1.set_zlim(0, 1)
    ax1.zaxis.set_major_locator(LinearLocator(10))
    ax1.zaxis.set_major_formatter('{x:.02f}')
    ax1.view_init(elev=25., azim=-140)

    # Optionally draw the theoretical boundary curve when smoothing is enabled
    if USE_SMOOTH_EDGE:
        sigma_curve = sigma_values_f1_dense
        c_edge_curve = 2.0 * (1.0 - sigma_curve)  # exact boundary
        valid = (c_edge_curve > 0)
        c_edge_curve = c_edge_curve[valid]
        sigma_curve = sigma_curve[valid]
        rho_edge_curve, _ = rho_f1_closed_masked(c_edge_curve, sigma_curve, margin_rel=0.0)
        ax1.plot(np.log10(c_edge_curve), sigma_curve, rho_edge_curve, color='k', linewidth=1.4, label='_nolegend_')

    # f2 surface
    surf2 = ax2.plot_surface(C2_log, S2, rho_values_f2, cmap=cm.viridis, linewidth=0, antialiased=True)
    ax2.set_title('Ours: ' + r'$f2(\rho,\sigma,c)=(1-\sigma-\rho)\exp\!\left(-\frac{2(1-\sigma-\rho)}{c}\right)-\rho$', color='r')
    ax2.set_xlabel('c (log scale)', color='#00008B')
    ax2.set_ylabel(r'$\sigma$', color='#00008B')
    ax2.set_zlabel('Maximum value of ' + r'$\rho$', color='#00008B')
    ax2.set_zlim(0, 1)
    ax2.zaxis.set_major_locator(LinearLocator(10))
    ax2.zaxis.set_major_formatter('{x:.02f}')
    ax2.view_init(elev=25., azim=-140)

    # Shared x-ticks (in linear space) rendered on log10(c) axis
    xticks = [1, 2, 4, 10, 30, 60, 100]
    ax1.set_xticks(np.log10(xticks)); ax1.set_xticklabels(xticks); ax1.tick_params(axis='x', rotation=-35)
    ax2.set_xticks(np.log10(xticks)); ax2.set_xticklabels(xticks); ax2.tick_params(axis='x', rotation=-35)

    # Common colorbar (ignores NaNs)
    combined = np.concatenate((rho_values_f1.ravel(), rho_values_f2.ravel()))
    finite = np.isfinite(combined)
    vmin, vmax = (combined[finite].min(), combined[finite].max()) if finite.any() else (0.0, 1.0)

    fig.subplots_adjust(bottom=0.1)
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    mappable = cm.ScalarMappable(cmap=cm.viridis)
    mappable.set_clim(vmin, vmax)
    fig.colorbar(mappable, cax=cbar_ax, orientation='horizontal')

    plt.show()


if __name__ == "__main__":
    main()
