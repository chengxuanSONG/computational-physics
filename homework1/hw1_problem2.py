import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
# 2. Output directory
# ============================================================

output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)


# ============================================================
# 3. Function
# ============================================================

def f(t):
    """
    f(t) = exp(-t), evaluated in single precision.
    """
    t = np.asarray(t, dtype=np.float32)
    return np.exp(-t).astype(np.float32)


# ============================================================
# 4. Midpoint rule
# ============================================================

def midpoint_rule(N):
    """
    Composite midpoint rule using N bins.

        I ≈ h * sum f(a + (i + 1/2) h)

    Global truncation error: O(h^2) = O(N^-2)
    """

    a = np.float32(0.0)
    b = np.float32(1.0)
    N32 = np.float32(N)

    h = np.float32((b - a) / N32)

    i = np.arange(N, dtype=np.float32)

    x_mid = a + (i + np.float32(0.5)) * h

    values = f(x_mid)

    # Force accumulation in float32
    total = np.sum(values, dtype=np.float32)

    return np.float32(h * total)


# ============================================================
# 5. Trapezoid rule
# ============================================================

def trapezoid_rule(N):
    """
    Composite trapezoid rule using N bins.

        I ≈ h [1/2 f(a)
               + sum f(a + i h)
               + 1/2 f(b)]

    Global truncation error: O(h^2) = O(N^-2)
    """

    a = np.float32(0.0)
    b = np.float32(1.0)
    N32 = np.float32(N)

    h = np.float32((b - a) / N32)

    if N > 1:
        i = np.arange(1, N, dtype=np.float32)
        x = a + i * h
        interior_sum = np.sum(f(x), dtype=np.float32)
    else:
        interior_sum = np.float32(0.0)

    endpoint_sum = np.float32(
        np.float32(0.5) * f(a)
        + np.float32(0.5) * f(b)
    )

    total = np.float32(endpoint_sum + interior_sum)

    return np.float32(h * total)


# ============================================================
# 6. Simpson's rule
# ============================================================

def simpson_rule(N):
    """
    Composite Simpson's rule.

    N must be even.

        I ≈ h/3 [
            f(a) + f(b)
            + 4 sum f(x_odd)
            + 2 sum f(x_even)
        ]

    Global truncation error: O(h^4) = O(N^-4)
    """

    if N % 2 != 0:
        raise ValueError("Simpson's rule requires even N.")

    a = np.float32(0.0)
    b = np.float32(1.0)
    N32 = np.float32(N)

    h = np.float32((b - a) / N32)

    # Odd indices: 1, 3, 5, ...
    odd_i = np.arange(1, N, 2, dtype=np.float32)
    x_odd = a + odd_i * h

    # Even indices: 2, 4, 6, ...
    even_i = np.arange(2, N, 2, dtype=np.float32)
    x_even = a + even_i * h

    odd_sum = np.sum(f(x_odd), dtype=np.float32)
    even_sum = np.sum(f(x_even), dtype=np.float32)

    total = np.float32(
        f(a)
        + f(b)
        + np.float32(4.0) * odd_sum
        + np.float32(2.0) * even_sum
    )

    return np.float32(
        (h / np.float32(3.0)) * total
    )


# ============================================================
# 7. Exact value
# ============================================================

def exact_integral():
    """
    Exact value:
        ∫_0^1 exp(-t) dt = 1 - exp(-1)

    Evaluated in float64 for use as a reference.
    """
    return 1.0 - np.exp(-1.0)


# ============================================================
# 8. Relative error
# ============================================================

def relative_error(numerical, exact):
    return abs(
        (np.float64(numerical) - np.float64(exact))
        / np.float64(exact)
    )


# ============================================================
# 9. Compute errors
# ============================================================

def calculate_errors(N_values):

    exact = exact_integral()

    midpoint_errors = []
    trapezoid_errors = []
    simpson_errors = []

    for N in N_values:

        I_mid = midpoint_rule(N)
        I_trap = trapezoid_rule(N)
        I_simp = simpson_rule(N)

        midpoint_errors.append(
            relative_error(I_mid, exact)
        )

        trapezoid_errors.append(
            relative_error(I_trap, exact)
        )

        simpson_errors.append(
            relative_error(I_simp, exact)
        )

    return (
        np.asarray(midpoint_errors),
        np.asarray(trapezoid_errors),
        np.asarray(simpson_errors),
        exact
    )


# ============================================================
# 10. Find minimum error
# ============================================================

def print_minimum_error(name, N_values, errors):

    valid = np.isfinite(errors) & (errors > 0)

    if not np.any(valid):
        print(f"{name:15s}: no valid values")
        return

    valid_indices = np.where(valid)[0]

    local_index = np.argmin(errors[valid])
    index = valid_indices[local_index]

    print(
        f"{name:15s}: "
        f"minimum relative error = {errors[index]:.6e}, "
        f"optimal N = {N_values[index]}"
    )


# ============================================================
# 11. Plot
# ============================================================

def make_plot():

    # Powers of two are natural for a log-log convergence plot.
    #
    # N = 2, 4, 8, ..., 2^20
    #
    # The large values are included deliberately so that
    # single-precision roundoff effects become visible.
    N_values = np.array(
        [2**k for k in range(1, 21)],
        dtype=np.int64
    )

    (
        err_mid,
        err_trap,
        err_simp,
        exact
    ) = calculate_errors(N_values)

    fig, ax = plt.subplots(figsize=(8.2, 6.4))

    ax.loglog(
        N_values,
        err_mid,
        marker="o",
        markersize=5,
        linewidth=2.2,
        label="Midpoint rule"
    )

    ax.loglog(
        N_values,
        err_trap,
        marker="s",
        markersize=5,
        linewidth=2.2,
        linestyle="--",
        label="Trapezoid rule"
    )

    ax.loglog(
        N_values,
        err_simp,
        marker="^",
        markersize=5,
        linewidth=2.2,
        linestyle="-.",
        label="Simpson's rule"
    )

    ax.set_xlabel(r"Number of bins $N$")
    ax.set_ylabel(r"Relative error $\epsilon$")

    ax.set_title(
        r"Numerical integration of "
        r"$\int_0^1 e^{-t}\,dt$"
    )

    ax.grid(
        True,
        which="major",
        linestyle=":",
        linewidth=0.8,
        alpha=0.55
    )

    ax.grid(
        True,
        which="minor",
        linestyle=":",
        linewidth=0.45,
        alpha=0.25
    )

    ax.legend(loc="best")

    png_name = output_dir / "integration_error.png"
    pdf_name = output_dir / "integration_error.pdf"

    fig.savefig(png_name, dpi=300)
    fig.savefig(pdf_name)

    plt.close(fig)

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "Exact integral = "
        f"{exact:.12e}"
    )

    print(
        "float32 machine epsilon = "
        f"{np.finfo(np.float32).eps:.12e}"
    )

    print("-" * 70)

    print_minimum_error(
        "Midpoint",
        N_values,
        err_mid
    )

    print_minimum_error(
        "Trapezoid",
        N_values,
        err_trap
    )

    print_minimum_error(
        "Simpson",
        N_values,
        err_simp
    )

    print("\nSaved:")
    print(png_name)
    print(pdf_name)

    print("\nDetailed results:")
    print(
        f"{'N':>10s} "
        f"{'Midpoint':>14s} "
        f"{'Trapezoid':>14s} "
        f"{'Simpson':>14s}"
    )

    for N, em, et, es in zip(
        N_values,
        err_mid,
        err_trap,
        err_simp
    ):
        print(
            f"{N:10d} "
            f"{em:14.6e} "
            f"{et:14.6e} "
            f"{es:14.6e}"
        )


# ============================================================
# 12. Main
# ============================================================

if __name__ == "__main__":

    print("Machine epsilon information:")
    print(
        "float32 epsilon =",
        np.finfo(np.float32).eps
    )

    make_plot()

    print("\nAll calculations completed.")