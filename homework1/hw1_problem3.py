import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline
from scipy.integrate import simpson
from scipy.signal import find_peaks
from pathlib import Path


# ============================================================
# 1. Plot style
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


# ============================================================
# 2. Input/output
# ============================================================

data_file = "lcdm_z0.matter_pk"

output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)


# ============================================================
# 3. Load tabulated power spectrum
# ============================================================

data = np.loadtxt(data_file)

# Homework says:
# column 1 = k
# column 2 = P(k)
# ignore remaining columns
k_data = data[:, 0]
P_data = data[:, 1]

# Only use positive values for log-log interpolation
valid = (k_data > 0.0) & (P_data > 0.0)

k_data = k_data[valid]
P_data = P_data[valid]

print("Loaded power spectrum:")
print(f"Number of valid points = {len(k_data)}")
print(f"k_min = {k_data.min():.6e} h/Mpc")
print(f"k_max = {k_data.max():.6e} h/Mpc")


# ============================================================
# 4. Log-log cubic spline interpolation
# ============================================================

# The data span many orders of magnitude and are sampled
# approximately logarithmically, so interpolate log P vs log k.

logk_data = np.log(k_data)
logP_data = np.log(P_data)

logP_spline = CubicSpline(
    logk_data,
    logP_data,
    bc_type="natural"
)


def power_spectrum(k):
    """
    Evaluate P(k) using a cubic spline in log(k)-log(P).

    Parameters
    ----------
    k : array-like
        Wavenumber in h/Mpc.

    Returns
    -------
    P : array-like
        Interpolated power spectrum.
    """
    k = np.asarray(k, dtype=np.float64)

    return np.exp(
        logP_spline(np.log(k))
    )


# ============================================================
# 5. Correlation function
# ============================================================

def xi_of_r(r, kmax=10.0, nk=100000):
    r"""
    Compute

        xi(r) = 1/(2*pi^2)
                integral dk k^2 P(k) sin(kr)/(kr)

    using Simpson integration on a dense linear k-grid.

    Parameters
    ----------
    r : float
        Separation in Mpc/h.

    kmax : float
        Upper integration limit in h/Mpc.

    nk : int
        Number of grid points used in the k integral.

    Returns
    -------
    xi : float
        Correlation function xi(r).
    """

    kmin = k_data.min()

    # Do not integrate beyond the supplied data range
    kmax = min(kmax, k_data.max())

    # Dense linear grid is useful because sin(kr) is oscillatory
    k = np.linspace(
        kmin,
        kmax,
        nk,
        dtype=np.float64
    )

    P = power_spectrum(k)

    kr = k * r

    # sin(kr)/(kr)
    kernel = np.sinc(kr / np.pi)

    integrand = (
        k**2
        * P
        * kernel
    )

    integral = simpson(
        integrand,
        x=k
    )

    return integral / (2.0 * np.pi**2)


# ============================================================
# 6. Compute xi(r)
# ============================================================

# Fine enough r grid to resolve the BAO bump
r_values = np.linspace(
    50.0,
    120.0,
    281
)

# Main integration cutoff
kmax_main = 10.0

print("\nComputing correlation function...")

xi_values = np.array([
    xi_of_r(
        r,
        kmax=kmax_main
    )
    for r in r_values
])

r2xi = r_values**2 * xi_values


# ============================================================
# 7. Find BAO local peak
# ============================================================

# The BAO bump should lie on large scales near ~100 Mpc/h.
# Restrict peak finding so that the high value at r=50 is
# not incorrectly interpreted as the BAO peak.

bao_min = 80.0
bao_max = 120.0

bao_mask = (
    (r_values >= bao_min)
    &
    (r_values <= bao_max)
)

r_bao = r_values[bao_mask]
y_bao = r2xi[bao_mask]

# Find actual local maxima, not the global endpoint maximum
peak_indices, properties = find_peaks(
    y_bao
)

if len(peak_indices) == 0:
    raise RuntimeError(
        "No local BAO peak was found between "
        f"{bao_min} and {bao_max} Mpc/h."
    )

# If more than one local maximum exists, choose the largest one
best_peak_index = peak_indices[
    np.argmax(y_bao[peak_indices])
]

r_peak_grid = r_bao[best_peak_index]
peak_grid_value = y_bao[best_peak_index]


# ============================================================
# 8. Refine peak position with a local cubic spline
# ============================================================

# Use several points around the detected peak to obtain a
# smoother estimate of the peak position.

i0 = max(best_peak_index - 3, 0)
i1 = min(best_peak_index + 4, len(r_bao))

r_local = r_bao[i0:i1]
y_local = y_bao[i0:i1]

peak_spline = CubicSpline(
    r_local,
    y_local
)

r_fine = np.linspace(
    r_local.min(),
    r_local.max(),
    5000
)

y_fine = peak_spline(r_fine)

fine_index = np.argmax(y_fine)

r_peak = r_fine[fine_index]
peak_value = y_fine[fine_index]

print("\nMain result:")
print(
    f"Grid BAO peak = "
    f"{r_peak_grid:.3f} Mpc/h"
)

print(
    f"Refined BAO peak = "
    f"{r_peak:.3f} Mpc/h"
)

print(
    f"Peak r^2 xi(r) = "
    f"{peak_value:.6e}"
)


# ============================================================
# 9. Robustness check versus k_max
# ============================================================

kmax_values = [
    1.0,
    2.0,
    5.0,
    10.0
]


def find_bao_peak(r, y):
    """
    Find the strongest local maximum in the BAO interval.
    """

    mask = (
        (r >= bao_min)
        &
        (r <= bao_max)
    )

    rr = r[mask]
    yy = y[mask]

    peaks, _ = find_peaks(yy)

    if len(peaks) == 0:
        return np.nan

    p = peaks[np.argmax(yy[peaks])]

    # Refine locally
    j0 = max(p - 3, 0)
    j1 = min(p + 4, len(rr))

    spline_local = CubicSpline(
        rr[j0:j1],
        yy[j0:j1]
    )

    rr_fine = np.linspace(
        rr[j0:j1].min(),
        rr[j0:j1].max(),
        3000
    )

    yy_fine = spline_local(rr_fine)

    return rr_fine[np.argmax(yy_fine)]


print("\nRobustness check:")

robustness_results = []

for kmax in kmax_values:

    print(
        f"  Computing k_max = {kmax:g} ..."
    )

    xi_test = np.array([
        xi_of_r(
            r,
            kmax=kmax
        )
        for r in r_values
    ])

    r2xi_test = (
        r_values**2
        * xi_test
    )

    test_peak = find_bao_peak(
        r_values,
        r2xi_test
    )

    robustness_results.append(
        (kmax, test_peak)
    )

    print(
        f"    BAO peak = "
        f"{test_peak:.3f} Mpc/h"
    )


# ============================================================
# 10. Main BAO plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.2, 6.4)
)

ax.plot(
    r_values,
    r2xi,
    linewidth=2.5,
    label=(
        rf"$k_{{\max}}="
        rf"{kmax_main:g}\,h/\mathrm{{Mpc}}$"
    )
)

# Vertical line at BAO peak
ax.axvline(
    r_peak,
    linestyle="--",
    linewidth=1.6,
    alpha=0.8
)

# Peak marker
ax.scatter(
    [r_peak],
    [peak_value],
    s=65,
    zorder=5
)

# Annotation
ax.annotate(
    (
        rf"$r_{{\rm BAO}}"
        rf"\simeq {r_peak:.1f}\,"
        rf"\mathrm{{Mpc}}/h$"
    ),
    xy=(
        r_peak,
        peak_value
    ),
    xytext=(
        r_peak - 23.0,
        peak_value * 1.08
    ),
    fontsize=14,
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.1
    )
)

ax.set_xlabel(
    r"Separation $r\;[\mathrm{Mpc}/h]$"
)

ax.set_ylabel(
    r"$r^2\xi(r)$"
)

ax.set_title(
    r"BAO feature in the matter correlation function"
)

ax.set_xlim(
    50.0,
    120.0
)

ax.grid(
    True,
    linestyle=":",
    linewidth=0.7,
    alpha=0.5
)

ax.legend(
    loc="best"
)

png_name = (
    output_dir
    / "bao_correlation.png"
)

pdf_name = (
    output_dir
    / "bao_correlation.pdf"
)

fig.savefig(
    png_name,
    dpi=300
)

fig.savefig(
    pdf_name
)

plt.close(fig)


# ============================================================
# 11. Optional robustness plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.2, 6.4)
)

for kmax in kmax_values:

    xi_test = np.array([
        xi_of_r(
            r,
            kmax=kmax
        )
        for r in r_values
    ])

    y_test = (
        r_values**2
        * xi_test
    )

    ax.plot(
        r_values,
        y_test,
        linewidth=1.8,
        label=(
            rf"$k_{{\max}}="
            rf"{kmax:g}$"
        )
    )

ax.set_xlabel(
    r"Separation $r\;[\mathrm{Mpc}/h]$"
)

ax.set_ylabel(
    r"$r^2\xi(r)$"
)

ax.set_title(
    r"Dependence on the upper integration limit"
)

ax.set_xlim(
    80.0,
    120.0
)

ax.grid(
    True,
    linestyle=":",
    linewidth=0.7,
    alpha=0.5
)

ax.legend(
    title=r"$k_{\max}\ [h/\mathrm{Mpc}]$"
)

robust_png = (
    output_dir
    / "bao_kmax_robustness.png"
)

robust_pdf = (
    output_dir
    / "bao_kmax_robustness.pdf"
)

fig.savefig(
    robust_png,
    dpi=300
)

fig.savefig(
    robust_pdf
)

plt.close(fig)


# ============================================================
# 12. Final output
# ============================================================

print("\nRobustness summary:")

for kmax, peak in robustness_results:
    print(
        f"k_max = {kmax:4.1f} h/Mpc"
        f"   r_BAO = {peak:.3f} Mpc/h"
    )

print("\nSaved:")
print(png_name)
print(pdf_name)
print(robust_png)
print(robust_pdf)

print("\nAll calculations completed.")