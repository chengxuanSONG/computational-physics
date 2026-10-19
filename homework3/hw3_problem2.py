"""Newman 7.4: smooth Dow closing values with Fourier low-pass filters.

Run from any directory. Requires NumPy and Matplotlib.
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


def low_pass(coefficients, fraction):
    """Keep the first floor(fraction * len(coefficients)) FFT entries.

    This count includes the DC term. Return a copy so the input spectrum
    remains available for independently applying the other cutoff.
    """
    if not 0 < fraction <= 1:
        raise ValueError("The retained fraction must be in (0, 1].")
    keep = max(1, int(fraction * len(coefficients)))
    filtered = coefficients.copy()
    filtered[keep:] = 0.0
    return filtered, keep


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ========================================================
    # 2. Parts (a)-(b): read the data and calculate the real FFT
    # ========================================================
    dow = np.loadtxt(BASE_DIR / "dow.txt")
    if dow.ndim != 1 or len(dow) < 2 or not np.isfinite(dow).all():
        raise ValueError("Expected a finite one-dimensional series of closing values.")
    n = len(dow)
    # The data contain successive business days, not calendar dates.
    days = np.arange(n)
    coefficients = np.fft.rfft(dow)

    # ========================================================
    # 3. Parts (c)-(e): discard high frequencies and invert
    # ========================================================
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                             sharey=True, layout="constrained")
    lines = [
        "Newman Exercise 7.4: Fourier filtering and smoothing",
        f"Number of business-day observations: {n}",
        f"Number of rfft coefficients: {len(coefficients)}",
        "Retention counts use floor(fraction * number of rfft coefficients).",
        "The retained count includes the DC term at k=0.",
        f"Original mean closing value: {dow.mean():.8f}",
    ]
    for ax, fraction, color in zip(axes, (0.10, 0.02), ("tab:orange", "tab:red")):
        filtered, keep = low_pass(coefficients, fraction)
        smoothed = np.fft.irfft(filtered, n=n)
        rms_residual = np.sqrt(np.mean((dow - smoothed)**2))
        shortest_period = n / (keep - 1) if keep > 1 else np.inf

        ax.plot(days, dow, color="0.55", linewidth=0.8, label="Original daily data")
        ax.plot(days, smoothed, color=color, linewidth=2,
                label=f"First {fraction:.0%} retained ({keep} coefficients)")
        ax.set_title(f"Fourier smoothing: retain first {fraction:.0%}")
        ax.set_ylabel("Dow Jones closing value")
        ax.set_xlim(0, n-1)
        ax.grid(linestyle=":", alpha=0.4)
        ax.legend(loc="lower left")

        lines.extend([
            "",
            f"Retained fraction requested: {fraction:.0%}",
            f"Retained entries: {keep} / {len(coefficients)} ({keep/len(coefficients):.6%})",
            f"Retained indices: k = 0 through {keep-1}",
            f"Highest retained frequency: {(keep-1)/n:.8f} cycles/business-day sample",
            f"Shortest retained period: {shortest_period:.8f} business-day samples",
            f"Smoothed mean closing value: {smoothed.mean():.8f}",
            f"RMS removed component: {rms_residual:.8f} index points",
        ])
    axes[-1].set_xlabel("Business-day index")

    for extension in ("png", "pdf"):
        path = OUTPUT_DIR / f"dow_fourier_smoothing.{extension}"
        fig.savefig(path)
        print(f"Saved: {path}")
    plt.close(fig)
    summary = "\n".join(lines) + "\n"
    (BASE_DIR / "hw3_problem2_results.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    main()
