from parameters import *
from linear_algebra.tdma import tdma
import numpy as np


def solve(T, F_new, F_old):
    """
    Расчёт температуры в новых координатах на новом шаге по времени с помощью схемы расщепления.
    T – значения температуры на сетке в НОВЫХ координатах
    F_new – положение границы фазового перехода на n+1 шаге по времени
    F_old – положение границы фазового перехода на n-ом шаге по времени
    dx – шаг по x на сетке в новых координатах
    dy – шаг по y на сетке в новых координатах
    """
    F_new = np.copy(F_new)
    F_old = np.copy(F_old)
    inv_dx = 1.0/dx
    inv_dy = 1.0/dy
    inv_dt = 1.0/dt

    temp_T = np.copy(T)

    a = c = np.ones((N_X - 1,)) * inv_dx * inv_dx * -dt
    b = np.ones((N_X - 1,)) * (1 + 2 * dt * inv_dx * inv_dx)

    a_y = np.empty((N_Y - 1),)
    b_y = np.empty((N_Y - 1),)
    c_y = np.empty((N_Y - 1),)

    alpha_0 = 0.0  # Из левого граничного условия по y (1-го рода)
    beta_0 = T_ice/T_0  # Из левого граничного условия по y

    # РАСЧЕТ ПО Y
    for j in range(1, N_X - 1):
        inv_F_new = 1.0/F_new[j]
        for k in range(0, N_Y - 1):
            y = k*dy
            kappa_k = y * inv_F_new * (inv_dt * (F_new[j] - F_old[j]) +
                                       2 * (0.5 * inv_dx * (F_new[j + 1] - F_new[j - 1]))**2 -
                                       inv_dx * inv_dx * (F_new[j + 1] - 2 * F_new[j] + F_new[j - 1]))
            sigma_k = W * W * inv_F_new * inv_F_new + \
                      (y * 0.5 * inv_dx * inv_F_new * (F_new[j + 1] - F_new[j - 1])) ** 2
            a_y[k] = -dt * (0.5 * inv_dy * kappa_k + inv_dy * inv_dy * sigma_k)
            b_y[k] = 1 + dt * 2 * inv_dy * inv_dy * sigma_k
            c_y[k] = dt * (0.5 * inv_dy * kappa_k - inv_dy * inv_dy * sigma_k)

        temp_T[:, j] = tdma(
            alpha_0=alpha_0,
            beta_0=beta_0,
            condition_type=1,  # Граничное условие 1-го рода
            phi=T_0/T_0,  # На границе ф.п. T = T_0
            a=a_y,
            b=b_y,
            c=c_y,
            f=T[:, j]
        )

    alpha_0 = 1.0  # Из левого граничного условия по x (2-го рода)
    beta_0 = 0.0  # Из левого граничного условия по x

    # РАСЧЕТ ПО X
    for k in range(1, N_Y - 1):
        # ПРОГОНКА
        temp_T[k, :] = tdma(
            alpha_0=alpha_0,
            beta_0=beta_0,
            condition_type=2,
            phi=0.0,
            a=a,
            b=b,
            c=c,
            f=temp_T[k, :]
        )
    return temp_T
