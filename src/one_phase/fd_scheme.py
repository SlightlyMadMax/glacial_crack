from parameters import *
from linear_algebra.tdma import tdma
import numpy as np


def predict_correct(T, F_new, F_old, dx, dy):
    """
    Расчёт температуры в новых координатах на новом шаге по времени с помощью схемы предиктор-корректор.
    T – значения температуры на сетке в НОВЫХ координатах
    F_new – положение границы фазового перехода
    F_old – положение границы фазового перехода на предыдущем шаге по времени
    dx – шаг по x на сетке в новых координатах
    dy – шаг по y на сетке в новых координатах
    """

    temp_T = np.copy(T)
    new_T = np.copy(T)

    # ПРЕДСКАЗАНИЕ ПО X
    for k in range(1, N_Y - 1):
        alpha_0 = 0  # Из левого граничного условия по x
        beta_0 = T_ice / T_0  # Из левого граничного условия по x

        a = c = np.ones((N_X - 1,)) * (-dt / (2 * dx ** 2))
        b = np.ones((N_X - 1,)) * (1 + dt / (dx ** 2))

        # ПРОГОНКА
        temp_T[k, :] = tdma(
            alpha_0=alpha_0,
            beta_0=beta_0,
            u_r=T[k, N_X - 1],
            a=a,
            b=b,
            c=c,
            f=T[k, :]
        )

    # ПРЕДСКАЗАНИЕ ПО Y
    for j in range(1, N_X - 1):
        alpha_0 = 0  # Из левого граничного условия по y
        beta_0 = T_ice / T_0  # Из левого граничного условия по y
        a_y = np.empty((N_Y - 1), )
        b_y = np.empty((N_Y - 1), )
        c_y = np.empty((N_Y - 1), )

        for k in range(0, N_Y - 1):
            y = k*dy
            kappa_k = y*((F_new[j] - F_old[j])/dt + 2*((F_new[j + 1] - F_new[j - 1])/(2*dx))**2 -
                         (F_new[j + 1] - 2*F_new[j] + F_new[j - 1])/dx**2)/F_new[j]
            sigma_k = (1 / F_new[j]) ** 2 + (y*(F_new[j + 1] - F_new[j - 1])/(2 * dx * F_new[j])) ** 2
            a_y[k] = -dt * (kappa_k / (4 * dy) + sigma_k / (2 * dy ** 2))
            b_y[k] = 1 + dt * sigma_k / (dy ** 2)
            c_y[k] = dt * (kappa_k / (4 * dy) - sigma_k / (2 * dy ** 2))

        temp_T[:, j] = tdma(
            alpha_0=alpha_0,
            beta_0=beta_0,
            u_r=T[N_Y - 1, j],
            a=a_y,
            b=b_y,
            c=c_y,
            f=T[:, j]
        )

    # КОРРЕКЦИЯ
    for k in range(1, N_Y - 1):
        y = k*dy
        for j in range(1, N_X - 1):
            kappa_k = y * ((F_new[j] - F_old[j]) / dt + 2 * ((F_new[j + 1] - F_new[j - 1]) / (2 * dx)) ** 2 -
                           (F_new[j + 1] - 2 * F_new[j] + F_new[j - 1]) / dx ** 2) / F_new[j]
            sigma_k = (1 / F_new[j]) ** 2 + (y * (F_new[j + 1] - F_new[j - 1]) / (2 * dx * F_new[j])) ** 2
            m_1 = (temp_T[k, j + 1] - 2.0 * temp_T[k, j] + temp_T[k, j - 1]) / (dx ** 2)
            m_2 = sigma_k * (temp_T[k + 1, j] - 2.0 * temp_T[k, j] + temp_T[k - 1, j]) / (dy ** 2)
            m_3 = kappa_k * (temp_T[k + 1, j] - temp_T[k - 1, j]) / (2.0 * dy)
            m_4 = 2*y*(F_new[j + 1] - F_new[j - 1])*(temp_T[k+1, j+1] - temp_T[k+1, j-1] - temp_T[k-1, j+1] +
                                                     temp_T[k-1, j-1])/(F_new[j]*8*dx**2*dy)
            new_T[k, j] = T[k, j] + dt * (m_1 + m_2 + m_3 - m_4)

    return temp_T
