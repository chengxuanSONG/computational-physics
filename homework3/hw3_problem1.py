"""Newman 7.2: detect the solar cycle using a discrete Fourier transform.

Run from any directory; data and output paths are relative to this script.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. Plot style and paths
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 17,
    "legend.fontsize": 11,
    "axes.linewidth": 1.1,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "figures"


def save_figure(fig, name):
    for extension in ("png", "pdf"):
        path = OUTPUT_DIR / f"{name}.{extension}"
        fig.savefig(path)
        print(f"Saved: {path}")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = np.loadtxt(BASE_DIR / "sunspots.txt")
    if data.ndim != 2 or data.shape[1] != 2 or not np.isfinite(data).all():
        raise ValueError("Expected two finite columns: month and sunspot number.")
    months, sunspots = data.T
    if not np.allclose(np.diff(months), 1.0):
        raise ValueError("The FFT requires equally spaced monthly observations.")
    n = len(sunspots)

    # ========================================================
    # 2. Part (a): inspect the time series
    # ========================================================
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.8), layout="constrained")
    for ax in axes:
        ax.plot(months, sunspots, linewidth=0.85, color="tab:blue")
        ax.set_ylabel("Sunspot number")
        ax.set_xlabel("Months since January 1749")
        ax.grid(linestyle=":", alpha=0.4)
    axes[0].set_xlim(months[0], months[-1])
    axes[0].set_title("Monthly sunspot observations")
    axes[1].set_xlim(0, 600)
    axes[1].set_title("First 600 months: cycle spacing of roughly 130 months")
    save_figure(fig, "sunspots_time_series")

    # ========================================================
    # 3. Parts (b)-(c): Fourier power and dominant period
    # ========================================================
    # NumPy's forward transform is unnormalized:
    # c_k = sum_n y_n exp(-2 pi i k n / N).
    # rfft contains all independent coefficients for real data.
    coefficients = np.fft.rfft(sunspots)
    power = np.abs(coefficients)**2
    k = np.arange(len(power))
    # k=0 is the nonzero mean, not an oscillation. Retain it in
    # the calculated spectrum but exclude it from peak finding.
    peak_k = int(np.argmax(power[1:]) + 1)
    period_months = n / peak_k  # sampling interval = one month
    period_years = period_months / 12.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    axes[0].semilogy(k[1:], power[1:], linewidth=0.9)
    axes[0].set_xlim(1, k[-1])
    axes[0].set_title("Power spectrum (nonzero frequencies)")
    axes[1].plot(k[1:], power[1:], linewidth=1.2, marker=".", markersize=4)
    axes[1].set_xlim(0, 80)
    axes[1].set_ylim(bottom=0)
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].set_title("Low-frequency peak")
    axes[1].plot(peak_k, power[peak_k], "o", color="tab:red")
    axes[1].annotate(
        f"$k={peak_k}$\n$T={period_months:.2f}$ months",
        xy=(peak_k, power[peak_k]), xytext=(39, 0.76*power[peak_k]),
        arrowprops={"arrowstyle": "->", "color": "tab:red"},
    )
    for ax in axes:
        ax.set_xlabel(r"Fourier index $k$")
        ax.set_ylabel(r"Power $|c_k|^2$")
        ax.grid(linestyle=":", alpha=0.4)
    save_figure(fig, "sunspots_power_spectrum")

    summary = (
        "Newman Exercise 7.2: Detecting periodicity\n"
        f"Number of monthly observations: {n}\n"
        f"Month range: {months[0]:.0f} to {months[-1]:.0f}\n"
        "Visual estimate: approximately 130 months (about 11 years)\n"
        f"Largest nonzero-frequency peak: k = {peak_k}\n"
        f"Peak power |c_k|^2 = {power[peak_k]:.10e}\n"
        f"Period N/k = {period_months:.8f} months\n"
        f"Period in years = {period_years:.8f}\n"
        f"DC coefficient c_0 = {coefficients[0].real:.8f}\n"
        "DC is excluded from the displayed spectra and peak search.\n"
    )
    print("\n" + summary)
    (BASE_DIR / "hw3_problem1_results.txt").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
