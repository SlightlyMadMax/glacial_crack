from parameters import *
from linear_algebra.tdma import tdma
import numpy as np

factor = 1.0 / (N_Y * (N_Y + 1.0))


def find_rhs(T, F_new, F_old, j: int, i: int):
    y = (j + 1.0) * (2.0 * N_Y - j) * factor
    inv_F_new = 1.0/F_new[i]
    inv_dx = 1.0/dx
    inv_dt = 1.0/dt

    # Производная F по x
    df_dx = F_new[i + 1] - F_new[i - 1]

    h_j = 2.0 * (N_Y - j) * factor

    # Первая и смешанная производная T по y
    if j == 0:
        h_1 = 2.0 * (N_Y - 1.0) * factor

        d_y = ((T[1, i] - T[0, i])*(h_j + h_1)*(h_j + h_1) - (T[2, i] - T[0, i])*h_j*h_j)/(h_j*h_1*(h_j+h_1))

        d_xy = ((T[1, i+1]-T[0, i+1]-T[1, i-1]+T[0, i-1])*(h_j+h_1)*(h_j+h_1) +
                (T[2, i-1]-T[0, i-1]-T[2, i+1]+T[0, i+1])*h_j*h_j)/(h_j*h_1*(h_j+h_1)*dx)

    elif j == N_Y - 1:
        h_0 = 6.0 * factor
        h_1 = 4.0 * factor

        d_y = ((T[N_Y-1, i] - T[N_Y-2, i])*(h_1 + h_0)*(h_1 + h_0) + (T[N_Y-3, i] - T[N_Y-1, i])*h_1*h_1)/(h_0*h_1*(h_1+h_0))

        d_xy = ((T[N_Y-1, i+1]-T[N_Y-2, i+1]-T[N_Y-1, i-1]+T[N_Y-2, i-1])*(h_0+h_1)*(h_0+h_1) +
                (T[N_Y-3, i+1] - T[N_Y-1, i+1] - T[N_Y-3, i-1] + T[N_Y-1, i-1])*h_1*h_1)/(h_1*h_0*(h_1+h_0)*dx)
    else:
        h_0 = 2.0 * (N_Y - j + 1.0) * factor
        h_2 = 2.0 * (N_Y - j - 1.0) * factor

        d_y = (T[j + 1, i] - T[j - 1, i])/(h_j+h_0)

        d_xy = (T[j + 1, i + 1] - T[j + 1, i - 1] - T[j - 1, i + 1] + T[j - 1, i - 1]) / (dx * (h_0 + h_2))

    # Вторая производная T по x
    d_xx = T[j, i + 1] - 2.0 * T[j, i] + T[j, i - 1]

    kappa = y * inv_F_new * (inv_dt * (F_new[i] - F_old[i]) + 0.5 * inv_dx * inv_dx * df_dx * df_dx -
                             inv_dx * inv_dx * (F_new[i + 1] - 2 * F_new[i] + F_new[i - 1]))

    zeta = -2 * y * inv_F_new * df_dx

    return T[j, i] + \
        dt * (d_xx * inv_dx * inv_dx +
              0.5 * zeta * d_xy +
              0.5 * kappa * d_y)  # 0.5 ?


def solve(T, F_new, F_old):
    temp_T = np.copy(T)
    new_T = np.copy(T)
    F_new = np.copy(F_new)
    F_old = np.copy(F_old)

    inv_dx = 1.0 / dx

    a_y = np.empty((N_Y - 1), )
    b_y = np.empty((N_Y - 1), )
    c_y = np.empty((N_Y - 1), )

    alpha_0 = 0.0  # Из левого граничного условия по y (1-го рода)
    beta_0 = T_ice/T_0  # Из левого граничного условия по y

    rhs = np.empty(N_Y,)
    for i in range(1, N_X - 1):
        inv_F_new = 1.0 / F_new[i]
        for j in range(0, N_Y - 1):
            y = (j + 1.0)*(2.0 * N_Y - j) * factor
            df_dx = F_new[i + 1] - F_new[i - 1]
            h_j = 2.0 * (N_Y - j) * factor
            h_0 = 2.0 * (N_Y - j + 1.0) * factor

            sigma_j = W * W * inv_F_new * inv_F_new + (0.5 * y * inv_dx * inv_F_new * df_dx)*(0.5 * y * inv_dx * inv_F_new * df_dx)

            a_y[j] = -2.0 * dt * sigma_j / (h_j * (h_j + h_0))
            c_y[j] = -2.0 * dt * sigma_j / (h_0 * (h_j + h_0))
            b_y[j] = 1.0 + 2.0 * dt * sigma_j / (h_j * h_0)

            rhs[j] = find_rhs(T, F_new, F_old, j, i)

        rhs[N_Y-1] = find_rhs(T, F_new, F_old, N_Y-1, i)

        # ПРОГОНКА
        temp_T[:, i] = tdma(
            alpha_0=alpha_0,
            beta_0=beta_0,
            condition_type=1,
            phi=1.0,  # T_0/T_0
            a=a_y,
            b=b_y,
            c=c_y,
            f=rhs
        )

    a = c = np.ones((N_X - 1,)) * inv_dx * inv_dx * -dt
    b = np.ones((N_X - 1,)) * (1.0 + 2.0 * dt * inv_dx * inv_dx)

    alpha_0 = 1.0  # Из левого граничного условия по x (2-го рода)
    beta_0 = 0.0  # Из левого граничного условия по x

    rhs = np.empty(N_X,)
    for j in range(1, N_Y - 1):
        for i in range(0, N_X):
            if i == 0:
                d_xx = T[j, 2] - 2.0 * T[j, 1] + T[j, 0]
            elif i == N_X - 1:
                d_xx = T[j, N_X - 1] - 2.0 * T[j, N_X - 2] + T[j, N_X - 3]
            else:
                d_xx = T[j, i + 1] - 2.0 * T[j, i] + T[j, i - 1]

            rhs[i] = temp_T[j, i] - dt * inv_dx * inv_dx * d_xx

        new_T[j, :] = tdma(
            alpha_0=alpha_0,
            beta_0=beta_0,
            condition_type=2,  # Граничное условие 2-го рода
            phi=0.0,
            a=a,
            b=b,
            c=c,
            f=rhs
        )
    return new_T
