import time

from parameters import *
from linear_algebra.tdma import tdma
import numpy as np
import numba


# @numba.jit
def find_rhs(T, F_new, F_old, j: int, i: int):
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_dt = 1.0 / dt
    j_int = int(0.5 * (N_Y - 1))
    chi = 1.0 if j < j_int else c_ice * rho_ice * k_w / (c_w * rho_w * k_ice)
    inv_F = 1.0 / F_new[i]
    inv_FH = 1.0 / (H - F_new[i])

    # Производная F по x
    df_dx = F_new[i + 1] - F_new[i - 1]

    # Первая производная T по y
    if j == 0:
        d_y = 4.0 * T[1, i] - 3.0 * T[0, i] - T[2, i]
    elif j == N_Y - 1:
        d_y = 3.0 * T[N_Y - 1, i] - 4.0 * T[N_Y - 2, i] + T[N_Y - 3, i]
    else:
        d_y = T[j + 1, i] - T[j - 1, i]

    # Вторая производная T по x
    d_xx = T[j, i + 1] - 2.0 * T[j, i] + T[j, i - 1]

    # Смешанная производная T
    if j == 0:
        d_xy = 4.0 * T[1, i + 1] - 3.0 * T[0, i + 1] - T[2, i + 1] - 4.0 * T[1, i - 1] + 3.0 * T[0, i - 1] + T[2, i - 1]
    elif j == N_Y - 1:
        d_xy = 3.0 * T[N_Y - 1, i + 1] - 4.0 * T[N_Y - 2, i + 1] + T[N_Y - 3, i + 1] - 3.0 * T[N_Y - 1, i - 1] + \
               4 * T[N_Y - 2, i - 1] - T[N_Y - 3, i - 1]
    else:
        d_xy = T[j + 1, i + 1] - T[j + 1, i - 1] - T[j - 1, i + 1] + T[j - 1, i - 1]

    if j < j_int:
        kappa = j * dy * inv_F * (inv_dt * (F_new[i] - F_old[i]) + 0.5 * inv_F * inv_dx * inv_dx * df_dx * df_dx -
                                  inv_dx * inv_dx * (F_new[i + 1] - 2 * F_new[i] + F_new[i - 1]))
        zeta = -j * dy * inv_F * inv_dx * df_dx
    else:
        kappa = inv_FH * (inv_dt * (2.0 - j * dy) * (F_new[i] - F_old[i]) +
                          0.5 * inv_FH * (j * dy - 1.0) * inv_dx * inv_dx * df_dx * df_dx -
                          (j * dy - 1.0) * inv_dx * inv_dx * (F_new[i + 1] - 2 * F_new[i] + F_new[i - 1]))
        zeta = (j * dy - 1.0) * inv_FH * inv_dx * df_dx

    return T[j, i] + \
           dt * (inv_dx * inv_dx * d_xx +
                 0.5 * zeta * inv_dx * inv_dy * d_xy +
                 0.5 * kappa * inv_dy * d_y) / chi


def solve(T, F_new, F_old):
    temp_T = np.copy(T)
    new_T = np.copy(T)
    F_new = np.copy(F_new)
    F_old = np.copy(F_old)
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    j_int = int(0.5 * (N_Y - 1))

    a_y = np.empty((N_Y - 1), )
    b_y = np.empty((N_Y - 1), )
    c_y = np.empty((N_Y - 1), )

    rhs = np.empty(N_Y, )
    for i in range(1, N_X - 1):
        inv_F_new = 1.0 / F_new[i]
        inv_FH_new = 1.0 / (H - F_new[i])

        for j in range(0, N_Y - 1):
            chi = 1.0 if j <= j_int else c_ice * rho_ice * k_w / (c_w * rho_w * k_ice)
            df_dx = F_new[i + 1] - F_new[i - 1]
            if j <= j_int:
                sigma_j = W * W * inv_F_new * inv_F_new + \
                          (0.5 * j * dy * inv_dx * inv_F_new * df_dx) * (0.5 * j * dy * inv_dx * inv_F_new * df_dx)
            else:
                sigma_j = W * W * inv_FH_new * inv_FH_new + \
                          (0.5 * (1.0 - j * dy) * inv_dx * inv_FH_new * df_dx) * (
                                      0.5 * (1.0 - j * dy) * inv_dx * inv_FH_new * df_dx)

            a_y[j] = c_y[j] = -inv_dy * inv_dy * dt * sigma_j / chi
            b_y[j] = 1.0 + 2.0 * dt * inv_dy * inv_dy * sigma_j / chi

            rhs[j] = find_rhs(T, F_new, F_old, j, i)

        rhs[N_Y - 1] = find_rhs(T, F_new, F_old, N_Y - 1, i)
        rhs[j_int] = T_0 / T_0

        # ПРОГОНКА ДЛЯ ЛЬДА
        temp_T[0:j_int + 1, i] = tdma(
            alpha_0=0.0,  # Из левого граничного условия по y (1-го рода)
            beta_0=T_ice / T_0,  # Из левого граничного условия по y (1-го рода)
            condition_type=1,
            phi=T_0 / T_0,  # правое граничное условие
            a=a_y[0:j_int + 1],
            b=b_y[0:j_int + 1],
            c=c_y[0:j_int + 1],
            f=rhs[0:j_int + 1]
        )
        chi = c_ice * rho_ice * k_w / (c_w * rho_w * k_ice)
        sigma_j = W * W * inv_FH_new * inv_FH_new
        a_y[j_int] = c_y[j_int] = -inv_dy * inv_dy * dt * sigma_j / chi
        b_y[j_int] = 1.0 + 2.0 * dt * inv_dy * inv_dy * sigma_j / chi

        # ПРОГОНКА ДЛЯ ВОДЫ
        temp_T[j_int:N_Y, i] = tdma(
            alpha_0=0.0,  # Из левого граничного условия по y (1-го рода)
            beta_0=T_0 / T_0,  # Из левого граничного условия по y (1-го рода)
            condition_type=1,
            phi=T_air / T_0,  # правое граничное условие
            a=a_y[j_int:N_Y],
            b=b_y[j_int:N_Y],
            c=c_y[j_int:N_Y],
            f=rhs[j_int:N_Y]
        )

    rhs = np.empty(N_X, )
    for j in range(1, N_Y - 1):
        chi = 1.0 if j < j_int else c_ice * rho_ice * k_w / (c_w * rho_w * k_ice)
        a = c = np.ones((N_X - 1,)) * inv_dx * inv_dx * -dt / chi
        b = np.ones((N_X - 1,)) * (1.0 + 2.0 * dt * inv_dx * inv_dx / chi)
        for i in range(0, N_X):
            if i == 0:
                d_xx = T[j, 2] - 2.0 * T[j, 1] + T[j, 0]
            elif i == N_X - 1:
                d_xx = T[j, N_X - 1] - 2.0 * T[j, N_X - 2] + T[j, N_X - 3]
            else:
                d_xx = T[j, i + 1] - 2.0 * T[j, i] + T[j, i - 1]

            rhs[i] = temp_T[j, i] - dt * inv_dx * inv_dx * d_xx / chi

        new_T[j, :] = tdma(
            alpha_0=1.0,  # Из левого граничного условия по x (2-го рода)
            beta_0=0.0,  # Из левого граничного условия по x
            condition_type=2,  # Граничное условие 2-го рода
            phi=0.0,
            a=a,
            b=b,
            c=c,
            f=rhs
        )

    return new_T
