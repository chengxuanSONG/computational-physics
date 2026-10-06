import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 15,
    "axes.labelsize": 17,
    "axes.titlesize": 18,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


# ============================================================
# Paths
# ============================================================

data_file = "smf_cosmos.dat"

output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)


# ============================================================
# Numerical gradient
# ============================================================

def numerical_gradient(func, params, steps):
    """
    Central-difference numerical gradient.
    """

    params = np.asarray(params, dtype=float)
    steps = np.asarray(steps, dtype=float)

    grad = np.zeros_like(params)

    for i in range(len(params)):

        dp = np.zeros_like(params)
        dp[i] = steps[i]

        f_plus = func(params + dp)
        f_minus = func(params - dp)

        grad[i] = (
            f_plus - f_minus
        ) / (
            2.0 * steps[i]
        )

    return grad


# ============================================================
# Gradient descent with backtracking
# ============================================================

def gradient_descent(
    func,
    initial_params,
    derivative_steps,
    learning_rate=1.0e-3,
    tolerance=1.0e-6,
    max_iter=10000
):
    """
    Multidimensional gradient descent using numerical
    derivatives and adaptive backtracking of the step size.
    """

    params = np.asarray(
        initial_params,
        dtype=float
    )

    param_history = [params.copy()]
    func_history = [func(params)]

    for iteration in range(1, max_iter + 1):

        grad = numerical_gradient(
            func,
            params,
            derivative_steps
        )

        grad_norm = np.linalg.norm(grad)

        if grad_norm < tolerance:
            return (
                params,
                iteration - 1,
                np.asarray(param_history),
                np.asarray(func_history)
            )

        current_value = func_history[-1]

        eta = learning_rate

        # Backtracking line search
        while eta > 1.0e-12:

            candidate = (
                params - eta * grad
            )

            candidate_value = func(candidate)

            # Require an actual decrease
            if (
                np.isfinite(candidate_value)
                and candidate_value < current_value
            ):
                break

            eta *= 0.5

        if eta <= 1.0e-12:
            raise RuntimeError(
                "Gradient descent could not find "
                "a decreasing step."
            )

        params = candidate

        param_history.append(
            params.copy()
        )

        func_history.append(
            candidate_value
        )

        if np.linalg.norm(
            param_history[-1]
            - param_history[-2]
        ) < tolerance:
            return (
                params,
                iteration,
                np.asarray(param_history),
                np.asarray(func_history)
            )

    raise RuntimeError(
        "Gradient descent did not converge."
    )


# ============================================================
# Part 1: test function
# ============================================================

def test_function(params):

    x, y = params

    return (
        (x - 2.0)**2
        + (y - 2.0)**2
    )


def run_test_problem():

    initial = np.array([
        -2.0,
        5.0
    ])

    (
        best,
        iterations,
        param_history,
        f_history
    ) = gradient_descent(
        test_function,
        initial_params=initial,
        derivative_steps=np.array([
            1.0e-5,
            1.0e-5
        ]),
        learning_rate=0.15,
        tolerance=1.0e-10,
        max_iter=1000
    )

    print("Gradient descent test")
    print("----------------------------------------")
    print(f"initial point = {initial}")
    print(f"minimum = {best}")
    print(f"f_min = {test_function(best):.6e}")
    print(f"iterations = {iterations}")

    # Contour plot
    x_grid = np.linspace(
        -3.0,
        5.0,
        300
    )

    y_grid = np.linspace(
        -1.0,
        6.0,
        300
    )

    X, Y = np.meshgrid(
        x_grid,
        y_grid
    )

    Z = (
        (X - 2.0)**2
        + (Y - 2.0)**2
    )

    fig, ax = plt.subplots(
        figsize=(7.5, 6.2)
    )

    ax.contour(
        X,
        Y,
        Z,
        levels=20,
        linewidths=1.0
    )

    ax.plot(
        param_history[:, 0],
        param_history[:, 1],
        marker="o",
        markersize=4,
        linewidth=1.8,
        label="Gradient-descent path"
    )

    ax.scatter(
        [2.0],
        [2.0],
        marker="*",
        s=100,
        label="Exact minimum"
    )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    ax.set_title(
        r"Gradient descent on "
        r"$f(x,y)=(x-2)^2+(y-2)^2$"
    )

    ax.legend()

    fig.savefig(
        output_dir
        / "gradient_descent_test.pdf"
    )

    fig.savefig(
        output_dir
        / "gradient_descent_test.png"
    )

    plt.close(fig)

run_test_problem()
# ============================================================
# Load COSMOS stellar mass function
# ============================================================

data = np.loadtxt(data_file)

logM_data = data[:, 0]
n_data = data[:, 1]
sigma_data = data[:, 2]


# ============================================================
# Schechter function
# ============================================================

def schechter_function(
    logM,
    log_phi_star,
    log_M_star,
    alpha
):
    """
    Schechter function.

    Optimization parameters:
        log_phi_star = log10(phi_star)
        log_M_star   = log10(M_star)
        alpha        = low-mass slope
    """

    M = 10.0**logM

    phi_star = (
        10.0**log_phi_star
    )

    M_star = (
        10.0**log_M_star
    )

    ratio = (
        M / M_star
    )

    return (
        phi_star
        * ratio**(alpha + 1.0)
        * np.exp(-ratio)
        * np.log(10.0)
    )


# ============================================================
# Chi-square
# ============================================================

def chi_square(params):

    (
        log_phi_star,
        log_M_star,
        alpha
    ) = params

    model = schechter_function(
        logM_data,
        log_phi_star,
        log_M_star,
        alpha
    )

    return np.sum(
        (
            (n_data - model)
            / sigma_data
        )**2
    )


# ============================================================
# Fit from multiple initial guesses
# ============================================================

initial_guesses = [
    np.array([
        -2.5,
        10.8,
        -1.2
    ]),
    np.array([
        -3.0,
        11.0,
        -1.0
    ]),
    np.array([
        -2.8,
        11.2,
        -0.8
    ])
]

fit_results = []

print("\nSchechter-function fits")
print("----------------------------------------")

for j, initial in enumerate(
    initial_guesses,
    start=1
):

    (
        best,
        iterations,
        param_history,
        chi_history
    ) = gradient_descent(
        chi_square,
        initial_params=initial,
        derivative_steps=np.array([
            1.0e-5,
            1.0e-5,
            1.0e-5
        ]),
        learning_rate=1.0e-3,
        tolerance=1.0e-6,
        max_iter=10000
    )

    fit_results.append(
        (
            initial,
            best,
            iterations,
            param_history,
            chi_history
        )
    )

    print(f"\nStart {j}")
    print(f"initial = {initial}")
    print(
        "best parameters = "
        f"{best}"
    )
    print(
        f"chi^2_min = "
        f"{chi_history[-1]:.8f}"
    )
    print(
        f"iterations = "
        f"{iterations}"
    )


# ============================================================
# Choose first converged solution as main fit
# ============================================================

(
    main_initial,
    best_params,
    main_iterations,
    main_param_history,
    main_chi_history
) = fit_results[0]

(
    log_phi_star_best,
    log_M_star_best,
    alpha_best
) = best_params

phi_star_best = (
    10.0**log_phi_star_best
)

M_star_best = (
    10.0**log_M_star_best
)

chi2_best = (
    main_chi_history[-1]
)


print("\nBest-fit Schechter parameters")
print("----------------------------------------")

print(
    "log10(phi_star) = "
    f"{log_phi_star_best:.8f}"
)

print(
    "phi_star = "
    f"{phi_star_best:.8e}"
)

print(
    "log10(M_star) = "
    f"{log_M_star_best:.8f}"
)

print(
    "M_star = "
    f"{M_star_best:.8e}"
)

print(
    "alpha = "
    f"{alpha_best:.8f}"
)

print(
    "chi^2_min = "
    f"{chi2_best:.8f}"
)


# ============================================================
# Plot chi-square versus iteration
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.0, 6.2)
)

for j, result in enumerate(
    fit_results,
    start=1
):

    initial = result[0]
    chi_history = result[4]

    ax.semilogy(
        np.arange(
            len(chi_history)
        ),
        chi_history,
        linewidth=2.0,
        label=(
            rf"Start {j}: "
            rf"$({initial[0]:.1f},"
            rf"{initial[1]:.1f},"
            rf"{initial[2]:.1f})$"
        )
    )

ax.set_xlabel(
    "Iteration"
)

ax.set_ylabel(
    r"$\chi^2$"
)

ax.set_title(
    r"Gradient-descent convergence"
)

ax.grid(
    True,
    linestyle=":",
    alpha=0.5
)

ax.legend()

fig.savefig(
    output_dir
    / "schechter_chi2_convergence.pdf"
)

fig.savefig(
    output_dir
    / "schechter_chi2_convergence.png"
)

plt.close(fig)


# ============================================================
# Plot best fit versus data
# ============================================================

logM_plot = np.linspace(
    logM_data.min() - 0.05,
    logM_data.max() + 0.05,
    500
)

best_model = schechter_function(
    logM_plot,
    log_phi_star_best,
    log_M_star_best,
    alpha_best
)

fig, ax = plt.subplots(
    figsize=(8.0, 6.2)
)

ax.errorbar(
    10.0**logM_data,
    n_data,
    yerr=sigma_data,
    fmt="o",
    markersize=6,
    capsize=3,
    label="COSMOS data"
)

ax.plot(
    10.0**logM_plot,
    best_model,
    linewidth=2.3,
    label="Best-fit Schechter function"
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(
    r"$M_{\rm gal}$"
)

ax.set_ylabel(
    r"$n(M_{\rm gal})$"
)

ax.set_title(
    r"COSMOS stellar mass function"
)

ax.grid(
    True,
    which="both",
    linestyle=":",
    alpha=0.45
)

ax.legend()

fig.savefig(
    output_dir
    / "schechter_best_fit.pdf"
)

fig.savefig(
    output_dir
    / "schechter_best_fit.png"
)

plt.close(fig)


print("\nSaved figures:")
print(
    output_dir
    / "gradient_descent_test.pdf"
)

print(
    output_dir
    / "schechter_chi2_convergence.pdf"
)

print(
    output_dir
    / "schechter_best_fit.pdf"
)

print("\nAll calculations completed.")