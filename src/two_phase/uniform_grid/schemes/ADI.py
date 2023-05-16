from parameters import *
from linear_algebra.tdma import tdma
import numpy as np
import numba


@numba.jit
def find_rhs(T, F_new, F_old, theta: float, j: int, i: int, chi: float):
    """
    :param T: матрица температур на текущем временном шаге
    :param F_new: одномерный массив, описывающий положение границы ф.п. на временном шаге n + 1
    :param F_old: одномерный массив, описывающий положение границы ф.п. на временном шаге n
    :param theta: численный параметр схемы переменных направлений
    :param j: шаг по координате y
    :param i: шаг по координате x
    :param chi: безразмерный коэффициент в уравнении теплопроводности, зависящий от среды
    :return: правая часть в первом шаге расщепленной схемы переменных направлений
    """
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_dt = 1.0 / dt
    j_int = int(0.5 * (N_Y - 1))  # индекс границы фазового перехода (y = 1.0)
    # chi = c_w * rho_w * k_ice / (c_ice * rho_ice * k_w)  # безразмерный параметр в уравнении теплопроводности
    inv_F = 1.0 / F_new[i]
    inv_FH = 1.0 / (H - F_new[i])

    # Производная F по x
    df_dx = F_new[i + 1] - F_new[i - 1]

    # Вторая производная T по x
    d_xx = T[j, i + 1] - 2.0 * T[j, i] + T[j, i - 1]

    # Производные температуры по y + смешанные
    if j == 0 or j == j_int + 1:
        d_y = 4.0 * T[1, i] - 3.0 * T[0, i] - T[2, i]
        d_yy = T[2, i] - 2.0 * T[1, i] + T[0, i]
        d_xy = 4.0 * T[1, i + 1] - 3.0 * T[0, i + 1] - T[2, i + 1] - 4.0 * T[1, i - 1] + 3.0 * T[0, i - 1] + T[2, i - 1]
    elif j == N_Y - 1 or j == j_int:
        d_y = 3.0 * T[j, i] - 4.0 * T[j - 1, i] + T[j - 2, i]
        d_yy = T[j, i] - 2.0 * T[j - 1, i] + T[j - 2, i]
        d_xy = 3.0 * T[j, i + 1] - 4.0 * T[j - 1, i + 1] + T[j - 2, i + 1] - 3.0 * T[j, i - 1] + \
               4 * T[j - 1, i - 1] - T[j - 2, i - 1]
    else:
        d_y = T[j + 1, i] - T[j - 1, i]
        d_yy = T[j + 1, i] - 2.0 * T[j, i] + T[j - 1, i]
        d_xy = T[j + 1, i + 1] - T[j + 1, i - 1] - T[j - 1, i + 1] + T[j - 1, i - 1]

    # Коэффициенты при производных
    if j <= j_int:
        kappa = j * dy * inv_F * (inv_dt * (F_new[i] - F_old[i]) +
                                  0.5 * inv_F * inv_dx * inv_dx * df_dx * df_dx -
                                  inv_dx * inv_dx * (F_new[i + 1] - 2.0 * F_new[i] + F_new[i - 1]))
        zeta = -j * dy * inv_F * inv_dx * df_dx
        sigma = W * W * inv_F * inv_F + \
                (0.25 * j * dy * j * dy * inv_F * inv_F * inv_dx * df_dx * inv_dx * df_dx)
    else:
        kappa = inv_FH * (j * dy - 2.0) * (0.5 * inv_FH * inv_dx * inv_dx * df_dx * df_dx / chi +
                                           inv_dx * inv_dx * (F_new[i + 1] - 2 * F_new[i] + F_new[i - 1]) / chi -
                                           inv_dt * (F_new[i] - F_old[i]))
        zeta = (j * dy - 2.0) * inv_FH * inv_dx * df_dx / chi
        sigma = W * W * inv_FH * inv_FH + \
                ((2.0 - j * dy) * inv_FH * 0.5 * inv_dx * df_dx) * ((2.0 - j * dy) * inv_FH * 0.5 * inv_dx * df_dx) / chi

    return T[j, i] + dt * (inv_dx * inv_dx * d_xx / chi +
                           (1.0 - theta) * sigma * inv_dy * inv_dy * d_yy +
                           0.5 * zeta * inv_dx * inv_dy * d_xy +
                           0.5 * kappa * inv_dy * d_y)


def solve(T, F_new, F_old, theta: float):
    """
    Решает уравнение методом переменных направлений
    :param T: матрица температур на текущем временном шаге
    :param F_new: вектор, описывающий положение границы ф.п. на временном шаге n + 1
    :param F_old: вектор, описывающий положение границы ф.п. на временном шаге n
    :param theta: численный параметр схемы переменных направлений
    :return: матрица температур на новом шаге по времени
    """
    temp_T = np.copy(T)
    new_T = np.copy(T)
    F_new = np.copy(F_new)
    F_old = np.copy(F_old)
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    j_int = int(0.5 * (N_Y - 1))  # индекс границы фазового перехода (y = 1.0)

    # массивы с прогоночными коэффициентами, значения задаются далее в цикле
    a_y = np.empty((N_Y - 1), )
    b_y = np.empty((N_Y - 1), )
    c_y = np.empty((N_Y - 1), )

    rhs = np.empty(N_Y, )
    for i in range(1, N_X - 1):
        inv_F = 1.0 / F_new[i]
        inv_FH = 1.0 / (H - F_new[i])
        df_dx = F_new[i + 1] - F_new[i - 1]
        for j in range(0, N_Y - 1):
            chi = 1.0 if j <= j_int else c_w * rho_w * k_ice / (c_ice * rho_ice * k_w)  # безразмерный параметр в уравнении теплопроводности
            if j <= j_int:
                sigma_j = W * W * inv_F * inv_F + \
                          (0.25 * j * dy * j * dy * inv_F * inv_F * inv_dx * df_dx * inv_dx * df_dx)
            else:
                sigma_j = W * W * inv_FH * inv_FH + \
                          0.25 * (2.0 - j * dy) * (2.0 - j * dy) * inv_FH * inv_dx * df_dx * inv_FH * inv_dx * df_dx

            # заполняем значения прогоночных коэффициентов
            a_y[j] = c_y[j] = -inv_dy * inv_dy * dt * theta * sigma_j / chi
            b_y[j] = 1.0 + 2.0 * dt * inv_dy * inv_dy * theta * sigma_j / chi
            # вычисляем правую часть для первого шага метода переменных направлений
            rhs[j] = find_rhs(T, F_new, F_old, theta, j, i, chi)

        chi = c_w * rho_w * k_ice / (c_ice * rho_ice * k_w)
        rhs[N_Y - 1] = find_rhs(T, F_new, F_old, theta, N_Y - 1, i, chi)

        # ПРОГОНКА ДЛЯ ЛЬДА (первый шаг метода переменных направлений)
        temp_T[0:j_int + 1, i] = tdma(
            alpha_0=0.0,  # Из левого граничного условия по y (1-го рода)
            beta_0=T_ice / T_0,  # Из левого граничного условия по y (1-го рода)
            condition_type=1,
            phi=T_0 / T_0,  # правое граничное условие
            a=a_y[0:j_int],
            b=b_y[0:j_int],
            c=c_y[0:j_int],
            f=rhs[0:j_int + 1]
        )

        rhs[j_int] = find_rhs(T, F_new, F_old, theta, j_int, i, chi)
        # ПРОГОНКА ДЛЯ ВОДЫ (первый шаг метода переменных направлений)
        temp_T[j_int:N_Y, i] = tdma(
            alpha_0=0.0,  # Из левого граничного условия по y (1-го рода)
            beta_0=T_0 / T_0,  # Из левого граничного условия по y (1-го рода)
            condition_type=1,
            phi=T_air / T_0,  # правое граничное условие
            a=a_y[j_int:N_Y-1],
            b=b_y[j_int:N_Y-1],
            c=c_y[j_int:N_Y-1],
            f=rhs[j_int:N_Y]
        )

    # второй шаг метода переменных направлений
    rhs = np.empty(N_X, )
    for j in range(1, N_Y - 1):
        chi = 1.0 if j <= j_int else c_w * rho_w * k_ice / (c_ice * rho_ice * k_w)  # безразмерный параметр в уравнении теплопроводности
        # массивы с прогоночными коэффициентами заполняем сразу же
        a = c = np.ones((N_X - 1,)) * inv_dx * inv_dx * -dt * theta / chi
        b = np.ones((N_X - 1,)) * (1.0 + 2.0 * dt * inv_dx * inv_dx * theta / chi)
        # вычисляем правую часть для второго шага метода ADI
        for i in range(0, N_X):
            if i == 0:
                d_xx = T[j, 2] - 2.0 * T[j, 1] + T[j, 0]
            elif i == N_X - 1:
                d_xx = T[j, N_X - 1] - 2.0 * T[j, N_X - 2] + T[j, N_X - 3]
            else:
                d_xx = T[j, i + 1] - 2.0 * T[j, i] + T[j, i - 1]

            rhs[i] = temp_T[j, i] - dt * theta * inv_dx * inv_dx * d_xx / chi

        # прогонка по X для второго шага
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
