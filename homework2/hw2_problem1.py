import math


# ============================================================
# Problem 1: Relaxation and overrelaxation
# Equation:
#
#     x = 1 - exp(-c x)
#
# with c = 2
# ============================================================

C = 2.0
TOL = 1.0e-6
X0 = 1.0
MAX_ITER = 10000


def f(x, c=C):
    """
    Fixed-point function:
        f(x) = 1 - exp(-c x)
    """
    return 1.0 - math.exp(-c * x)


def df(x, c=C):
    """
    Derivative:
        f'(x) = c exp(-c x)
    """
    return c * math.exp(-c * x)


# ============================================================
# Ordinary relaxation
# ============================================================

def ordinary_relaxation(x0=X0, c=C, tol=TOL):
    """
    Ordinary relaxation:
        x' = f(x)

    The error estimate is obtained from the usual relaxation
    formula.
    """

    x = x0

    for iteration in range(1, MAX_ITER + 1):

        x_new = f(x, c)

        fp = df(x, c)

        error_estimate = abs(
            (x - x_new) /
            (1.0 - 1.0 / fp)
        )

        if error_estimate < tol:
            return x_new, iteration, error_estimate

        x = x_new

    raise RuntimeError("Ordinary relaxation did not converge.")


# ============================================================
# Overrelaxation
# ============================================================

def overrelaxation(omega, x0=X0, c=C, tol=TOL):
    """
    Overrelaxation:

        x' = (1 + omega) f(x) - omega x

    Error estimate derived in Exercise 6.11(a):

        epsilon'
        ~= (x - x')
           /
           [1 - 1 / ((1+omega) f'(x) - omega)]
    """

    x = x0

    for iteration in range(1, MAX_ITER + 1):

        x_new = (
            (1.0 + omega) * f(x, c)
            - omega * x
        )

        gprime = (
            (1.0 + omega) * df(x, c)
            - omega
        )

        error_estimate = abs(
            (x - x_new) /
            (1.0 - 1.0 / gprime)
        )

        if error_estimate < tol:
            return x_new, iteration, error_estimate

        x = x_new

    raise RuntimeError(
        f"Overrelaxation did not converge for omega={omega}."
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("Equation: x = 1 - exp(-2x)")
    print(f"Required accuracy: {TOL:.1e}")

    # --------------------------------------------------------
    # Part (b): ordinary relaxation
    # --------------------------------------------------------

    x_relax, n_relax, err_relax = ordinary_relaxation()

    print("\nOrdinary relaxation:")
    print(f"solution = {x_relax:.10f}")
    print(f"iterations = {n_relax}")
    print(f"estimated error = {err_relax:.3e}")

    # --------------------------------------------------------
    # Part (c): overrelaxation for several omega values
    # --------------------------------------------------------

    print("\nOverrelaxation:")
    print(
        f"{'omega':>8s}"
        f"{'solution':>16s}"
        f"{'iterations':>14s}"
        f"{'error':>14s}"
    )

    omega_values = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.65,
        0.68,
        0.70,
        0.75
    ]

    for omega in omega_values:

        x_over, n_over, err_over = overrelaxation(
            omega
        )

        print(
            f"{omega:8.2f}"
            f"{x_over:16.10f}"
            f"{n_over:14d}"
            f"{err_over:14.3e}"
        )

    # --------------------------------------------------------
    # Approximate theoretical optimal omega
    # --------------------------------------------------------

    x_star = x_relax

    fp_star = df(
        x_star,
        C
    )

    omega_opt = (
        fp_star /
        (1.0 - fp_star)
    )

    print("\nApproximate local convergence analysis:")
    print(f"f'(x*) = {fp_star:.6f}")
    print(f"omega_opt ~= {omega_opt:.6f}")