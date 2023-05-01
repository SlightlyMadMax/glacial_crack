import numpy as np
import numba


@numba.jit
def tdma(
        alpha_0: float, beta_0: float, phi: float,
        a, b, c, f, psi: float = 0.0,
        h: float = 0.0, condition_type: int = 1):
    """
    Трёхдиагональная прогонка.
    :param alpha_0: прогоночный коэффициент, определяемый из граничных условий
    :param beta_0: прогоночный коэффициент, определяемый из граничных условий
    :param phi: значение функции для граничных условий 1-го или 2-го рода
    :param a: нижняя диагональ
    :param b: главная диагональ
    :param c: верхняя диагональ
    :param f: правая часть
    :param psi: значение функции для условия 3-го рода
    :param h: шаг по координате (для условий 2-го и 3-го рода)
    :param condition_type: тип граничного условия (1, 2, 3)
    :return: решение СЛАУ с заданными условиями
    """
    n = len(f)
    alpha = np.zeros(n - 1)
    beta = np.zeros(n - 1)

    u = np.zeros(n)

    alpha[0] = alpha_0
    beta[0] = beta_0

    for j in range(1, n - 1):
        alpha[j] = -a[j] / (b[j] + c[j] * alpha[j - 1])
        beta[j] = (f[j] - c[j] * beta[j - 1]) / (b[j] + c[j] * alpha[j - 1])

    # Определение температуры из граничных условий на правой границе
    if condition_type == 1:
        u[n - 1] = phi
    elif condition_type == 2:
        u[n - 1] = (h*phi + beta[n - 2])/(1 - alpha[n - 2])
    else:
        u[n - 1] = (h*psi + beta[n - 2])/(1 - alpha[n - 2] - h*phi)

    for j in range(n - 2, -1, -1):
        u[j] = alpha[j] * u[j + 1] + beta[j]

    return u
