import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. Plot style: clean, paper-like
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
# 3. Functions evaluated in SINGLE PRECISION
# ============================================================

def f_cos(x):
    """
    cos(x) evaluated in float32.
    """
    x = np.float32(x)
    return np.float32(np.cos(x))


def f_exp(x):
    """
    exp(x) evaluated in float32.
    """
    x = np.float32(x)
    return np.float32(np.exp(x))


# ============================================================
# 4. Numerical differentiation methods
# ============================================================

def forward_difference(f, x, h):
    """
    Forward difference:
        f'(x) ≈ [f(x+h) - f(x)] / h

    Leading truncation error: O(h)
    """
    x = np.float32(x)
    h = np.float32(h)

    numerator = np.float32(
        f(np.float32(x + h)) - f(x)
    )

    return np.float32(numerator / h)


def central_difference(f, x, h):
    """
    Central difference:
        f'(x) ≈ [f(x+h) - f(x-h)] / (2h)

    Leading truncation error: O(h^2)
    """
    x = np.float32(x)
    h = np.float32(h)

    numerator = np.float32(
        f(np.float32(x + h))
        - f(np.float32(x - h))
    )

    denominator = np.float32(2.0) * h

    return np.float32(numerator / denominator)


def extrapolated_difference(f, x, h):
    """
    Richardson-extrapolated central difference:

        D_ext(h) = [4 D_c(h/2) - D_c(h)] / 3

    Leading truncation error: O(h^4)
    """
    x = np.float32(x)
    h = np.float32(h)

    half_h = np.float32(h / np.float32(2.0))

    d_h = central_difference(f, x, h)
    d_h2 = central_difference(f, x, half_h)

    numerator = np.float32(
        np.float32(4.0) * d_h2 - d_h
    )

    return np.float32(
        numerator / np.float32(3.0)
    )


# ============================================================
# 5. High-precision reference derivatives
# ============================================================

def exact_cos_derivative(x):
    """
    Reference value evaluated in float64.
    """
    return -np.sin(np.float64(x))


def exact_exp_derivative(x):
    """
    Reference value evaluated in float64.
    """
    return np.exp(np.float64(x))


# ============================================================
# 6. Relative error
# ============================================================

def relative_error(numerical, exact):
    return np.abs(
        (np.float64(numerical) - np.float64(exact))
        / np.float64(exact)
    )


# ============================================================
# 7. Compute errors for one function at one x
# ============================================================

def calculate_errors(f, exact_derivative, x, h_values):

    exact = exact_derivative(x)

    error_forward = []
    error_central = []
    error_extrapolated = []

    for h in h_values:

        h32 = np.float32(h)

        # Numerical derivatives
        df_forward = forward_difference(f, x, h32)
        df_central = central_difference(f, x, h32)
        df_extrapolated = extrapolated_difference(f, x, h32)

        # Relative errors
        error_forward.append(
            relative_error(df_forward, exact)
        )

        error_central.append(
            relative_error(df_central, exact)
        )

        error_extrapolated.append(
            relative_error(df_extrapolated, exact)
        )

    return (
        np.asarray(error_forward),
        np.asarray(error_central),
        np.asarray(error_extrapolated),
        exact
    )


# ============================================================
# 8. Plotting function
# ============================================================

def make_plot(
    f,
    exact_derivative,
    x,
    function_label,
    filename
):

    # Step sizes.
    #
    # This range is chosen so that we can see both:
    #   1. truncation-error-dominated behavior
    #   2. roundoff/cancellation-dominated behavior
    #
    h_values = np.logspace(
        0,
        -8,
        220
    )

    (
        err_forward,
        err_central,
        err_extrapolated,
        exact
    ) = calculate_errors(
        f,
        exact_derivative,
        x,
        h_values
    )

    fig, ax = plt.subplots(figsize=(8.2, 6.4))

    # --------------------------------------------------------
    # Plot numerical errors
    # --------------------------------------------------------

    ax.loglog(
        h_values,
        err_forward,
        linewidth=2.3,
        label="Forward difference"
    )

    ax.loglog(
        h_values,
        err_central,
        linewidth=2.3,
        linestyle="--",
        label="Central difference"
    )

    ax.loglog(
        h_values,
        err_extrapolated,
        linewidth=2.3,
        linestyle="-.",
        label="Extrapolated difference"
    )

    # --------------------------------------------------------
    # Axis labels and title
    # --------------------------------------------------------

    ax.set_xlabel(r"Step size $h$")
    ax.set_ylabel(r"Relative error $\epsilon$")

    ax.set_title(
        rf"$f(x)={function_label}$ at $x={x:g}$"
    )

    # Light grid, useful on a log-log plot
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

    # --------------------------------------------------------
    # Make small h appear on the right
    #
    # This is visually convenient because:
    # left  -> larger h -> truncation error
    # right -> smaller h -> roundoff error
    # --------------------------------------------------------

    ax.invert_xaxis()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    png_name = output_dir / f"{filename}.png"
    pdf_name = output_dir / f"{filename}.pdf"

    fig.savefig(png_name, dpi=300)
    fig.savefig(pdf_name)

    plt.close(fig)

    # --------------------------------------------------------
    # Print useful information
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print(f"Function: {function_label}")
    print(f"x = {x}")
    print(f"Exact derivative = {exact:.12e}")
    print("-" * 65)

    print_minimum_error(
        "Forward",
        h_values,
        err_forward
    )

    print_minimum_error(
        "Central",
        h_values,
        err_central
    )

    print_minimum_error(
        "Extrapolated",
        h_values,
        err_extrapolated
    )

    print(f"\nSaved: {png_name}")
    print(f"Saved: {pdf_name}")


# ============================================================
# 9. Find the minimum error and optimal h
# ============================================================

def print_minimum_error(name, h_values, errors):

    # Ignore NaN and infinite values if they appear for extremely
    # small h due to float32 limitations.
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
        f"optimal h = {h_values[index]:.6e}"
    )


# ============================================================
# 10. Main program
# ============================================================

if __name__ == "__main__":

    print("Machine epsilon information:")
    print(
        "float32 epsilon =",
        np.finfo(np.float32).eps
    )

    # cos(x), x = 0.1
    make_plot(
        f_cos,
        exact_cos_derivative,
        x=0.1,
        function_label=r"\cos x",
        filename="cos_x_0p1"
    )

    # cos(x), x = 10
    make_plot(
        f_cos,
        exact_cos_derivative,
        x=10.0,
        function_label=r"\cos x",
        filename="cos_x_10"
    )

    # exp(x), x = 0.1
    make_plot(
        f_exp,
        exact_exp_derivative,
        x=0.1,
        function_label=r"e^x",
        filename="exp_x_0p1"
    )

    # exp(x), x = 10
    make_plot(
        f_exp,
        exact_exp_derivative,
        x=10.0,
        function_label=r"e^x",
        filename="exp_x_10"
    )

    print("\nAll calculations completed.")