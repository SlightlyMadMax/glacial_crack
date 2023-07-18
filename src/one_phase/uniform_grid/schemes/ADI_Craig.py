from parameters import *
from linear_algebra.tdma import tdma
import numpy as np


def time_centered_md(T_new, T_old, F_new, F_old, j: int, i: int):
    """
    Вычисляет смешанную производную и производную первого порядка от разницы температур
    на первом и втором шаге схемы для увеличения порядка точности по времени (Крейг и Снейд 1988)
    :param T_new: матрица температур на текущей итерации схемы переменных направлений
    :param T_old: матрица температур на предыдущей итерации схемы переменных направлений
    :param F_new: одномерный массив, описывающий положение границы ф.п. на временном шаге n + 1
    :param F_old: одномерный массив, описывающий положение границы ф.п. на временном шаге n
    :param j: шаг по координате y
    :param i: шаг по координате x
    :return: слагаемое, добавляемое на второй итерации метода ADI
    """
    T = T_new - T_old
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_dt = 1.0 / dt
    inv_F_new = 1.0 / F_new[i]

    # Производная F по x
    df_dx = F_new[i + 1] - F_new[i - 1]

    # Смешанные производные и производные первого порядка
    if j == 0:
        d_y = 4.0 * T[1, i] - 3.0 * T[0, i] - T[2, i]
        d_xy = 4.0 * T[1, i + 1] - 3.0 * T[0, i + 1] - T[2, i + 1] - 4.0 * T[1, i - 1] + 3.0 * T[0, i - 1] + T[2, i - 1]
    elif j == N_Y - 1:
        d_y = 3.0 * T[N_Y - 1, i] - 4.0 * T[N_Y - 2, i] + T[N_Y - 3, i]
        d_xy = 3.0 * T[N_Y - 1, i + 1] - 4.0 * T[N_Y - 2, i + 1] + T[N_Y - 3, i + 1] - 3.0 * T[N_Y - 1, i - 1] + \
               4 * T[N_Y - 2, i - 1] - T[N_Y - 3, i - 1]
    else:
        d_y = T[j + 1, i] - T[j - 1, i]
        d_xy = T[j + 1, i + 1] - T[j + 1, i - 1] - T[j - 1, i + 1] + T[j - 1, i - 1]

    kappa = j * dy * inv_F_new * (inv_dt * (F_new[i] - F_old[i]) +
                                  0.5 * inv_F_new * inv_dx * df_dx * inv_dx * df_dx -
                                  inv_dx * inv_dx * (F_new[i + 1] - 2.0 * F_new[i] + F_new[i - 1]))

    zeta = -j * dy * inv_F_new * inv_dx * df_dx

    return dt * 0.5 * (zeta * d_xy * inv_dx * inv_dy + kappa * d_y * inv_dy)


def find_rhs(T, F_new, F_old, j: int, i: int):
    inv_F_new = 1.0 / F_new[i]
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_dt = 1.0 / dt

    # Производная F по x
    df_dx = F_new[i + 1] - F_new[i - 1]

    # Вторая производная T по x
    d_xx = T[j, i + 1] - 2.0 * T[j, i] + T[j, i - 1]

    # Производные по y и смешанные производные
    if j == 0:
        d_y = 4.0 * T[1, i] - 3.0 * T[0, i] - T[2, i]
        d_xy = 4.0 * T[1, i + 1] - 3.0 * T[0, i + 1] - T[2, i + 1] - 4.0 * T[1, i - 1] + 3.0 * T[0, i - 1] + T[2, i - 1]
    elif j == N_Y - 1:
        d_y = 3.0 * T[N_Y - 1, i] - 4.0 * T[N_Y - 2, i] + T[N_Y - 3, i]
        d_xy = 3.0 * T[N_Y - 1, i + 1] - 4.0 * T[N_Y - 2, i + 1] + T[N_Y - 3, i + 1] - 3.0 * T[N_Y - 1, i - 1] + \
               4 * T[N_Y - 2, i - 1] - T[N_Y - 3, i - 1]
    else:
        d_y = T[j + 1, i] - T[j - 1, i]
        d_xy = T[j + 1, i + 1] - T[j + 1, i - 1] - T[j - 1, i + 1] + T[j - 1, i - 1]

    kappa = j * dy * inv_F_new * (inv_dt * (F_new[i] - F_old[i]) +
                                  0.5 * inv_F_new * inv_dx * df_dx * inv_dx * df_dx -
                                  inv_dx * inv_dx * (F_new[i + 1] - 2.0 * F_new[i] + F_new[i - 1]))

    zeta = -j * dy * inv_F_new * df_dx * inv_dx

    return T[j, i] + \
        dt * (d_xx * inv_dx * inv_dx +
              0.25 * zeta * d_xy * inv_dx * inv_dy +
              0.5 * kappa * d_y * inv_dy)


def solve(T_old, F_new, F_old, T_new = None):
    temp_T = np.copy(T_old)
    new_T = np.copy(T_old)
    F_new = np.copy(F_new)
    F_old = np.copy(F_old)

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy

    a_y = np.empty((N_Y - 1), )
    b_y = np.empty((N_Y - 1), )
    c_y = np.empty((N_Y - 1), )

    rhs = np.empty(N_Y,)
    for i in range(1, N_X - 1):
        df_dx = F_new[i + 1] - F_new[i - 1]
        inv_F_new = 1.0 / F_new[i]
        for j in range(0, N_Y - 1):
            sigma_j = W * W * inv_F_new * inv_F_new + \
                      (0.25 * j * dy * inv_dx * inv_F_new * df_dx * j * dy * inv_dx * inv_F_new * df_dx)
            a_y[j] = c_y[j] = -dt * inv_dy * inv_dy * sigma_j
            b_y[j] = 1.0 + 2.0 * dt * inv_dy * inv_dy * sigma_j
            tc_term = 0.0
            if T_new is not None:
                tc_term = time_centered_md(T_new, T_old, F_new, F_old, j, i)
            rhs[j] = find_rhs(T_old, F_new, F_old, j, i) + tc_term

        tc_term = 0.0
        if T_new is not None:
            tc_term = time_centered_md(T_new, T_old, F_new, F_old, N_Y - 1, i)

        rhs[N_Y-1] = find_rhs(T_old, F_new, F_old, N_Y-1, i) + tc_term

        # ПРОГОНКА по y
        temp_T[:, i] = tdma(
            alpha_0=0.0,  # Из левого граничного условия по y (1-го рода)
            beta_0=T_ice / T_0,  # Из левого граничного условия по y
            condition_type=1,
            phi=1.0,  # T_0/T_0
            a=a_y,
            b=b_y,
            c=c_y,
            f=rhs
        )

    a = c = np.ones((N_X - 1,)) * -dt * inv_dx * inv_dx
    b = np.ones((N_X - 1,)) * (1.0 + 2.0 * dt * inv_dx * inv_dx)

    rhs = np.empty(N_X,)
    for j in range(1, N_Y - 1):
        for i in range(0, N_X):
            if i == 0:
                d_xx = T_old[j, 2] - 2.0 * T_old[j, 1] + T_old[j, 0]
            elif i == N_X - 1:
                d_xx = T_old[j, N_X - 1] - 2.0 * T_old[j, N_X - 2] + T_old[j, N_X - 3]
            else:
                d_xx = T_old[j, i + 1] - 2.0 * T_old[j, i] + T_old[j, i - 1]

            rhs[i] = temp_T[j, i] - dt * inv_dx * inv_dx * d_xx

        new_T[j, :] = tdma(
            alpha_0=1.0,  # Из левого граничного условия по x (2-го рода)
            beta_0=0.0,  # Из левого граничного условия по x
            condition_type=2,
            phi=0.0,
            a=a,
            b=b,
            c=c,
            f=rhs
        )

    return new_T
