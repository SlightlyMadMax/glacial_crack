from parameters import *
from linear_algebra.tdma import tdma
import numpy as np


def find_rhs(T, F_new, F_old, theta: float, j: int, i: int):
    inv_F_new = 1.0 / F_new[i]
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_dt = 1.0 / dt

    # Первая производная T по y
    d_y = T[j + 1, i] - T[j - 1, i]

    # Вторая производная T по y
    d_yy = T[j + 1, i] - 2.0 * T[j, i] + T[j - 1, i]

    # Производные по x
    if i == 0:
        d_xx = T[j, 2] - 2.0 * T[j, 1] + T[j, 0]
        d_xy = 4.0 * T[j + 1, 1] - 3.0 * T[j + 1, 0] - T[j + 1, 2] - 4.0 * T[j - 1, 1] + 3.0 * T[j - 1, 0] + T[j - 1, 2]
        df_dx = 4.0 * F_new[1] - 3.0 * F_new[0] - F_new[2]
        df_dxx = F_new[2] - 2.0 * F_new[1] + F_new[0]
    elif i == N_X - 1:
        d_xx = T[j, N_X - 1] - 2.0 * T[j, N_X - 2] + T[j, N_X - 3]
        d_xy = 3.0 * T[j + 1, N_X - 1] - 4.0 * T[j + 1, N_X - 2] + T[j + 1, N_X - 3] - 3.0 * T[j - 1, N_X - 1] + \
               4 * T[j - 1, N_X - 2] - T[j - 1, N_X - 3]
        df_dx = 3.0 * F_new[N_X - 1] - 4.0 * F_new[N_X - 2] + F_new[N_X - 3]
        df_dxx = F_new[N_X - 1] - 2.0 * F_new[N_X - 2] + F_new[N_X - 3]
    else:
        d_xx = T[j, i + 1] - 2.0 * T[j, i] + T[j, i - 1]
        d_xy = T[j + 1, i + 1] - T[j + 1, i - 1] - T[j - 1, i + 1] + T[j - 1, i - 1]
        df_dx = F_new[i + 1] - F_new[i - 1]
        df_dxx = F_new[i + 1] - 2.0 * F_new[i] + F_new[i - 1]

    sigma = W * W * inv_F_new * inv_F_new + \
            (0.25 * j * dy * j * dy * inv_dx * inv_dx * df_dx * df_dx * inv_F_new * inv_F_new)

    kappa = j * dy * inv_F_new * (inv_dt * (F_new[i] - F_old[i]) +
                                  0.5 * inv_F_new * inv_dx * df_dx * inv_dx * df_dx -
                                  inv_dx * inv_dx * df_dxx)

    zeta = -j * dy * inv_F_new * df_dx * inv_dx

    return T[j, i] + \
        dt * ((1.0 - theta) * d_xx * inv_dx * inv_dx +
              sigma * d_yy * inv_dy * inv_dy +
              0.5 * zeta * d_xy * inv_dx * inv_dy +
              0.5 * kappa * d_y * inv_dy)


def solve(T, F_new, F_old, theta: float):
    temp_T = np.copy(T)
    new_T = np.copy(T)
    F_new = np.copy(F_new)
    F_old = np.copy(F_old)

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy

    a = c = np.ones((N_X - 1,)) * -dt * inv_dx * inv_dx * theta
    b = np.ones((N_X - 1,)) * (1.0 + 2.0 * dt * inv_dx * inv_dx * theta)

    rhs = np.empty(N_X, )
    for j in range(1, N_Y - 1):
        for i in range(0, N_X):
            rhs[i] = find_rhs(T, F_new, F_old, theta, j, i)

        # ПРОГОНКА по x
        temp_T[j, :] = tdma(
            alpha_0=1.0,  # Из левого граничного условия по x (2-го рода)
            beta_0=0.0,  # Из левого граничного условия по x
            condition_type=2,  # Граничное условие 2-го рода
            phi=0.0,
            a=a,
            b=b,
            c=c,
            f=rhs
        )

    a_y = np.empty((N_Y - 1), )
    b_y = np.empty((N_Y - 1), )
    c_y = np.empty((N_Y - 1), )

    rhs = np.empty(N_Y,)
    for i in range(1, N_X - 1):
        df_dx = F_new[i + 1] - F_new[i - 1]
        inv_F_new = 1.0 / F_new[i]
        for j in range(0, N_Y - 1):
            if j == 0:
                d_yy = T[2, i] - 2.0 * T[1, i] + T[0, i]
            else:
                d_yy = T[j + 1, i] - 2.0 * T[j, i] + T[j - 1, i]
            sigma_j = W * W * inv_F_new * inv_F_new + \
                      (0.25 * j * dy * j * dy * inv_dx * inv_dx * df_dx * df_dx * inv_F_new * inv_F_new)
            a_y[j] = c_y[j] = -dt * inv_dy * inv_dy * theta * sigma_j
            b_y[j] = 1.0 + 2.0 * dt * inv_dy * inv_dy * theta * sigma_j
            rhs[j] = temp_T[j, i] - theta * sigma_j * dt * inv_dy * inv_dy * d_yy

        sigma_j = W * W * inv_F_new * inv_F_new + \
                  (0.25 * (N_Y - 1) * dy * (N_Y - 1) * dy * inv_dx * df_dx * inv_dx * df_dx * inv_F_new * inv_F_new)
        rhs[N_Y-1] = temp_T[N_Y - 1, i] - \
                     theta * sigma_j * dt * inv_dy * inv_dy * (T[N_Y - 1, i] - 2.0 * T[N_Y - 2, i] + T[N_Y - 3, i])

        # ПРОГОНКА по y
        new_T[:, i] = tdma(
            alpha_0=0.0,  # Из левого граничного условия по y (1-го рода)
            beta_0=T_ice / T_0,  # Из левого граничного условия по y
            condition_type=1,
            phi=1.0,  # T_0/T_0
            a=a_y,
            b=b_y,
            c=c_y,
            f=rhs
        )

    return new_T
