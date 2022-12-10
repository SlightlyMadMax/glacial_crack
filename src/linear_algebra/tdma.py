import numpy as np


def tdma(alpha_0, beta_0, u_r, a, b, c, f):
    """
    Tridiagonal matrix algorithm.
    Here alpha_0 and beta_0 are starting coefficients,
    a – lower diagonal, b – main diagonal, c – upper diagonal,
    f – right side vector
    """
    n = len(f)
    alpha = np.zeros(n - 1, float)
    beta = np.zeros(n - 1, float)

    u = np.zeros(n, float)

    alpha[0] = alpha_0
    beta[0] = beta_0

    u[n - 1] = u_r

    for j in range(1, n - 1):
        alpha[j] = -a[j] / (b[j] + c[j] * alpha[j - 1])
        beta[j] = (f[j] - c[j] * beta[j - 1]) / (b[j] + c[j] * alpha[j - 1])

    for j in range(n - 2, -1, -1):
        u[j] = alpha[j] * u[j + 1] + beta[j]

    return u
