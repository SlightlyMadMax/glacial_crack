from parameters import *
from linear_algebra.tdma import tdma
from schemes.utils.differences import *
import numpy as np


def find_rhs(T, F_new, F_old, theta: float, j: int, i: int):
    inv_F_new = 1.0/F_new[i]
    inv_dx = 1.0/dx
    inv_dt = 1.0/dt

    sigma = W * W * inv_F_new * inv_F_new + \
              (j*dy * 0.5 * inv_dx * inv_F_new * d_x(F_new, i)) ** 2

    kappa = j*dy * inv_F_new * (inv_dt * (F_new[i] - F_old[i]) +
                               2 * (0.5 * inv_dx * d_x(F_new, i)) ** 2 -
                               inv_dx * inv_dx * (F_new[i + 1] - 2 * F_new[i] + F_new[i - 1]))
    zeta = -2*j*dy*inv_F_new*d_x(F_new, i)

    rhs = T[j, i] + \
        dt * ((1.0 - theta)*sigma*d_yy(T, j, i) / (dy*dy) +
              d_xx(T, j, i) / (dx*dx) +
              0.5 * zeta * d_xy(T, j, i) / (dx*dy) +
              0.5 * kappa * d_y(T, j, i) / dy)
    return rhs


def solve(T, F_new, F_old):
    temp_T = np.copy(T)
    new_T = np.copy(T)
    F_new = np.copy(F_new)
    F_old = np.copy(F_old)

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy

    theta = 1.0

    a_y = np.empty((N_Y - 1), )
    b_y = np.empty((N_Y - 1), )
    c_y = np.empty((N_Y - 1), )

    alpha_0 = 0.0  # Из левого граничного условия по y (1-го рода)
    beta_0 = T_ice/T_0  # Из левого граничного условия по y

    rhs = np.empty(N_Y,)
    for i in range(1, N_X - 1):
        inv_F_new = 1.0 / F_new[i]
        for j in range(0, N_Y - 1):
            sigma_j = W * W * inv_F_new * inv_F_new + \
                          (j * dy * 0.5 * inv_dx * inv_F_new * (d_x(F_new, i))) ** 2
            a_y[j] = c_y[j] = inv_dy * inv_dy * -dt * theta * sigma_j
            b_y[j] = 1 + 2 * dt * inv_dy * inv_dy * theta * sigma_j
            rhs[j] = find_rhs(T, F_new, F_old, theta, j, i)
        rhs[N_Y-1] = find_rhs(T, F_new, F_old, theta, N_Y-1, i)

        # ПРОГОНКА
        temp_T[:, i] = tdma(
            alpha_0=alpha_0,
            beta_0=beta_0,
            condition_type=1,
            phi=T_0/T_0,
            a=a_y,
            b=b_y,
            c=c_y,
            f=rhs
        )

    a = c = np.ones((N_X - 1,)) * inv_dx * inv_dx * -dt * theta
    b = np.ones((N_X - 1,)) * (1 + 2 * dt * inv_dx * inv_dx * theta)

    alpha_0 = 1.0  # Из левого граничного условия по x (2-го рода)
    beta_0 = 0.0  # Из левого граничного условия по x

    rhs = np.empty(N_X,)
    for j in range(1, N_Y - 1):
        for i in range(0, N_X):
            rhs[i] = temp_T[j, i] - theta * dt * inv_dx * inv_dx * d_xx(T, j, i)
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
