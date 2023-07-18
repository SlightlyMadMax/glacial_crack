from parameters import *
from linear_algebra.tdma import tdma
import numpy as np


def find_rhs(T, F_new, F_old, f: float, j: int, i: int):
    inv_F_new = 1.0 / F_new[i]
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_dt = 1.0 / dt

    # Производная F по x
    df_dx = F_new[i + 1] - F_new[i - 1]

    # Вторая производная T по x
    d_xx = T[j, i + 1] - 2.0 * T[j, i] + T[j, i - 1]

    # Производные температуры по y + смешанные
    if j == 0:
        d_y = 4.0 * T[1, i] - 3.0 * T[0, i] - T[2, i]
        d_yy = T[2, i] - 2.0 * T[1, i] + T[0, i]
        d_xy = 4.0 * T[1, i + 1] - 3.0 * T[0, i + 1] - T[2, i + 1] - 4.0 * T[1, i - 1] + 3.0 * T[0, i - 1] + T[2, i - 1]
        d_xx_yy = (T[2, i + 1] - 2.0 * T[1, i + 1] + T[0, i + 1] - 2.0 * T[2, i] + 4.0 * T[1, i] -
                   2.0 * T[0, i] + T[2, i - 1] - 2.0 * T[1, i - 1] + T[0, i - 1])
    elif j == N_Y - 1:
        d_y = 3.0 * T[N_Y - 1, i] - 4.0 * T[N_Y - 2, i] + T[N_Y - 3, i]
        d_yy = T[N_Y - 1, i] - 2.0 * T[N_Y - 2, i] + T[N_Y - 3, i]
        d_xy = 3.0 * T[N_Y - 1, i + 1] - 4.0 * T[N_Y - 2, i + 1] + T[N_Y - 3, i + 1] - 3.0 * T[N_Y - 1, i - 1] + \
               4 * T[N_Y - 2, i - 1] - T[N_Y - 3, i - 1]
        d_xx_yy = (T[N_Y - 1, i + 1] - 2.0 * T[N_Y - 2, i + 1] + T[N_Y - 3, i + 1] - 2.0 * T[N_Y - 1, i] +
                   4.0 * T[N_Y - 2, i] - 2.0 * T[N_Y - 3, i] + T[N_Y - 1, i - 1] -
                   2.0 * T[N_Y - 2, i - 1] + T[N_Y - 3, i - 1])
    else:
        d_y = T[j + 1, i] - T[j - 1, i]
        d_yy = T[j + 1, i] - 2.0 * T[j, i] + T[j - 1, i]
        d_xy = T[j + 1, i + 1] - T[j + 1, i - 1] - T[j - 1, i + 1] + T[j - 1, i - 1]
        d_xx_yy = (T[j + 1, i + 1] - 2.0 * T[j, i + 1] + T[j - 1, i + 1] - 2.0 * T[j + 1, i] + 4.0 * T[j, i] - 2.0 * T[
            j - 1, i] +
                   T[j + 1, i - 1] - 2.0 * T[j, i - 1] + T[j - 1, i - 1])

    sigma = W * W * inv_F_new * inv_F_new + \
            (0.25 * j * dy * j * dy * inv_dx * inv_dx * df_dx * df_dx * inv_F_new * inv_F_new)

    kappa = j * dy * inv_F_new * (inv_dt * (F_new[i] - F_old[i]) +
                                  0.5 * inv_F_new * inv_dx * df_dx * inv_dx * df_dx -
                                  inv_dx * inv_dx * (F_new[i + 1] - 2 * F_new[i] + F_new[i - 1]))

    zeta = -j * dy * inv_F_new * df_dx * inv_dx

    return T[j, i] + \
          (1.0 / f + 0.5 * dt * sigma * inv_dy * inv_dy) * d_yy + \
          d_xx * dt * inv_dx * inv_dx + \
          0.25 * zeta * d_xy * dt * inv_dx * inv_dy + \
          0.5 * kappa * d_y * dt * inv_dy + \
          (1.0 + sigma) * d_xx_yy * dt * inv_dx * inv_dy


def solve(T, F_new, F_old, f: float):
    temp_T = np.copy(T)
    new_T = np.copy(T)
    F_new = np.copy(F_new)
    F_old = np.copy(F_old)

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy

    a_y = np.empty((N_Y - 1), )
    b_y = np.empty((N_Y - 1), )
    c_y = np.empty((N_Y - 1), )

    rhs = np.empty(N_Y, )
    for i in range(1, N_X - 1):
        df_dx = F_new[i + 1] - F_new[i - 1]
        inv_F_new = 1.0 / F_new[i]
        for j in range(0, N_Y - 1):
            sigma_j = W * W * inv_F_new * inv_F_new + \
                      (0.25 * j * dy * j * dy * inv_dx * inv_dx * df_dx * df_dx * inv_F_new * inv_F_new)
            a_y[j] = c_y[j] = 1.0 / f - 0.5 * dt * sigma_j * inv_dy * inv_dy
            b_y[j] = 1.0 - 2.0 * (1.0 / f - 0.5 * dt * sigma_j * inv_dy * inv_dy)
            rhs[j] = find_rhs(T, F_new, F_old, f, j, i)

        rhs[N_Y - 1] = find_rhs(T, F_new, F_old, f, N_Y - 1, i)

        # ПРОГОНКА
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

    a = c = np.ones((N_X - 1,)) * (1.0 / f - 0.5 * dt * inv_dx * inv_dx)
    b = np.ones((N_X - 1,)) * (1.0 - 2.0 * (1.0 / f - 0.5 * dt * inv_dx * inv_dx))

    rhs = np.empty(N_X, )
    for j in range(1, N_Y - 1):
        for i in range(0, N_X):
            if i == 0:
                d_xx = T[j, 2] - 2.0 * T[j, 1] + T[j, 0]
            elif i == N_X - 1:
                d_xx = T[j, N_X - 1] - 2.0 * T[j, N_X - 2] + T[j, N_X - 3]
            else:
                d_xx = T[j, i + 1] - 2.0 * T[j, i] + T[j, i - 1]

            rhs[i] = temp_T[j, i] + (1.0 / f - 0.5 * dt * inv_dx * inv_dx) * d_xx

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
