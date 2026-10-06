import math


# ============================================================
# Problem 2: Wien displacement constant
# ============================================================

# Physical constants (SI units)
h = 6.62607015e-34       # Planck constant [J s]
c = 299792458.0          # speed of light [m/s]
k_B = 1.380649e-23       # Boltzmann constant [J/K]

TOL = 1.0e-6


# ============================================================
# Nonlinear equation
#
#     5 exp(-x) + x - 5 = 0
#
# We want the nonzero positive root near x ~ 5.
# ============================================================

def f(x):
    return 5.0 * math.exp(-x) + x - 5.0


# ============================================================
# Binary search
# ============================================================

def binary_search(a, b, tol=TOL):
    """
    Solve f(x)=0 by binary search.

    The interval [a,b] must bracket a root.
    """

    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError("Initial interval does not bracket a root.")

    iterations = 0

    while (b - a) > tol:
        midpoint = 0.5 * (a + b)
        fm = f(midpoint)

        if fa * fm <= 0:
            b = midpoint
            fb = fm
        else:
            a = midpoint
            fa = fm

        iterations += 1

    root = 0.5 * (a + b)

    return root, iterations


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # Nonzero root lies between 4 and 5
    x_root, n_iter = binary_search(4.0, 5.0)

    # Wien displacement constant
    b_wien = h * c / (k_B * x_root)

    # Solar surface temperature from lambda_max = 502 nm
    lambda_sun = 502.0e-9

    T_sun = b_wien / lambda_sun

    print("Wien displacement calculation")
    print("----------------------------------------")
    print(f"x = {x_root:.10f}")
    print(f"iterations = {n_iter}")
    print(f"residual f(x) = {f(x_root):.3e}")

    print("\nWien displacement constant:")
    print(f"b = {b_wien:.10e} m K")

    print("\nSolar temperature estimate:")
    print(f"lambda_max = {lambda_sun:.3e} m")
    print(f"T = {T_sun:.2f} K")