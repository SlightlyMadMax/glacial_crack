import numpy as np


def tdma(alpha_0, beta_0, u_r, a, b, c, f):
    """
    Алгоритм прогонки.
    Здесь alpha_0 и beta_0 начальные прогоночные коэффициенты, определяемые из граничных условий
    u_r – последняя компонента решения, определяемая из граничных условий
    a – нижняя диагональ, b – главная диагональ, c – верхняя диагональ,
    f – вектор правой части
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
