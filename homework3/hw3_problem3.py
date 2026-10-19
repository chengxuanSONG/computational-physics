"""Newman 7.9: image deconvolution with a periodic Gaussian PSF.

Run from any directory. Requires NumPy and Matplotlib.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "figures"
SIGMA = 25.0  # pixels
EPSILON = 1.0e-3  # absolute threshold for the unnormalized PSF spectrum


def periodic_gaussian(shape, sigma):
    """Sample the exercise's unnormalized Gaussian with wrapped coordinates.

    The origin is at array index (0, 0); negative displacements are stored
    at the ends of each axis. No fftshift or normalization is applied.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    height, width = shape
    y = np.arange(height, dtype=float)
    x = np.arange(width, dtype=float)
    y[y > height // 2] -= height
    x[x > width // 2] -= width
    return np.exp(-(y[:, None]**2 + x[None, :]**2) / (2.0 * sigma**2))


def deconvolve(blurred, psf, epsilon=EPSILON):
    """Divide spectra only where |F| > epsilon, as the exercise suggests."""
    if blurred.shape != psf.shape:
        raise ValueError("Image and PSF must have the same shape.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    blurred_fft = np.fft.rfft2(blurred)
    psf_fft = np.fft.rfft2(psf)
    reliable = np.abs(psf_fft) > epsilon
    restored_fft = blurred_fft.copy()
    # Elsewhere retain the original blurred coefficient, rather than
    # dividing by a tiny number or setting it to zero.
    np.divide(blurred_fft, psf_fft, out=restored_fft, where=reliable)
    restored = np.fft.irfft2(restored_fft, s=blurred.shape)
    return restored, restored_fft, psf_fft, reliable


def save_figure(fig, name):
    for extension in ("png", "pdf"):
        path = OUTPUT_DIR / f"{name}.{extension}"
        fig.savefig(path)
        print(f"Saved: {path}")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    blurred = np.loadtxt(BASE_DIR / "blur.txt")
    if blurred.ndim != 2 or min(blurred.shape) < 2 or not np.isfinite(blurred).all():
        raise ValueError("Expected a finite two-dimensional brightness array.")
    psf = periodic_gaussian(blurred.shape, SIGMA)
    restored, restored_fft, psf_fft, reliable = deconvolve(blurred, psf)
    if not np.isfinite(restored).all():
        raise ValueError("The reconstruction contains non-finite values.")

    # Part (b): plot the array exactly as used in the transform. Bright
    # patches occupy the four corners because the Gaussian wraps around.
    fig, ax = plt.subplots(figsize=(6.3, 5.4), layout="constrained")
    im = ax.imshow(psf, cmap="gray", origin="upper", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_title(r"Periodic Gaussian point spread function ($\sigma=25$ pixels)")
    ax.set_xlabel("Column index")
    ax.set_ylabel("Row index")
    fig.colorbar(im, ax=ax, label="Point spread function")
    save_figure(fig, "image_point_spread_function")

    # Parts (a) and (c): row zero belongs at the top (bright sky).
    # Separate linear gray scales are necessary: the exercise uses an
    # unnormalized PSF, so input and recovered brightness scales differ.
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), layout="constrained")
    for ax, values, title in zip(
        axes, (blurred, restored), ("Original blurred image", "Deconvolved image")
    ):
        im = ax.imshow(values, cmap="gray", origin="upper",
                       interpolation="nearest", vmin=values.min(), vmax=values.max())
        ax.set_title(title)
        ax.set_xlabel("Column index")
        ax.set_ylabel("Row index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025, label="Brightness")
    save_figure(fig, "image_deconvolution")

    # Preserve full numerical precision in a compact file for reuse.
    np.save(BASE_DIR / "hw3_problem3_restored.npy", restored)
    # Check reblurring on the divided spectral bins, not on the fallback
    # bins, where the exercise deliberately does not invert the PSF.
    blurred_fft = np.fft.rfft2(blurred)
    spectral_residual = np.max(np.abs(
        restored_fft[reliable] * psf_fft[reliable] - blurred_fft[reliable]
    )) / np.max(np.abs(blurred_fft[reliable]))
    summary = (
        "Newman Exercise 7.9: Image deconvolution\n"
        f"Image shape: {blurred.shape}\n"
        f"Gaussian sigma: {SIGMA:.1f} pixels\n"
        f"Absolute Fourier threshold epsilon: {EPSILON:.1e}\n"
        f"Unnormalized PSF sum (DC coefficient): {psf.sum():.10f}\n"
        f"Stored rfft2 coefficient count: {reliable.size}\n"
        f"Coefficients divided: {np.count_nonzero(reliable)}\n"
        f"Coefficients left unchanged: {np.count_nonzero(~reliable)}\n"
        f"Fraction of stored coefficients divided: {reliable.mean():.8%}\n"
        f"Blurred brightness range: [{blurred.min():.8f}, {blurred.max():.8f}]\n"
        f"Restored brightness range: [{restored.min():.8f}, {restored.max():.8f}]\n"
        f"Mean blurred brightness: {blurred.mean():.10f}\n"
        f"Mean restored brightness: {restored.mean():.10f}\n"
        f"Relative spectral reblurring residual on divided bins: {spectral_residual:.3e}\n"
        "Display: origin='upper'; independent linear grayscale ranges; no clipping.\n"
    )
    (BASE_DIR / "hw3_problem3_results.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    main()
