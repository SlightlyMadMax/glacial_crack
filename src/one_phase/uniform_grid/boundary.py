import numpy as np
from parameters import *
import math


def init_f_vector(n_x):
    """
    Задание начального положения границы фазового перехода.
    :param n_x: число узлов по x
    :return: вектор с координатами границы фазового перехода
    """
    F = np.empty(n_x)
    for i in range(n_x):
        # Гладкая ступенька
        # F[:] = [0.2 + 0.5 / (1.0 + math.exp(-20.0 * (i * dx - 0.5))) for i in range(0, n_x)]

        # Плоскость
        # F[i] = a

        # Парабола f(x, t=0) = 3*(x - W/2)^2 + h
        # F[i] = a + 2*(i*dx - W/2)*(i*dx - W/2)

        # Трещина-гауссиана
        F[:] = [1.2 - 0.7 * math.exp(-(i * dx - 0.5) ** 2 / 0.01) for i in range(n_x)]

        # Подобие трещины
        # if i*dx < 0.4 or i*dx > 0.6:
        #     F[i] = 1.0
        # else:
        #     F[i] = 3.8 - 7.0*i*dx if i*dx < 0.5 else 7.0*i*dx - 3.2
    return F


def recalculate_boundary(F, T):
    """
    Пересчёт положения границы фазового перехода в соответствии с условием Стефана в новых координатах.
    :param F: вектор значений положения границы фазового перехода
    :param T: матрица значений температуры на сетке в НОВЫХ координатах
    :return: вектор с координатами границы фазового перехода
    """
    F_new = np.copy(F)
    inv_dy = 1.0 / dy
    inv_dx = 1.0 / dx
    inv_W = 1.0 / W
    inv_gamma = 1.0 / gamma

    for i in range(1, N_X-1):
        inv_F = 1.0/F[i]
        F_new[i] = F[i] + dt*inv_gamma*inv_F*(1.0 + (0.5*inv_W*inv_dx*(F[i+1]-F[i-1]))**2) *\
                   (0.5*inv_dy*(3.0 * T[N_Y - 1, i] - 4.0 * T[N_Y - 2, i] + T[N_Y - 3, i]))

    F_new[0] = F[0] + dt*inv_gamma*(1.0 + (0.5*inv_W*inv_dx*(4.0*F[1]-3.0*F[0]-F[2]))**2) *\
               (0.5*inv_dy*(3.0 * T[N_Y-1, 0] - 4.0 * T[N_Y-2, 0] + T[N_Y-3, 0]))/F[0]

    F_new[N_X-1] = F[N_X-1] + dt*inv_gamma*(1.0 + (0.5*inv_W*inv_dx*(3.0*F[N_X-1]-4.0*F[N_X-2]+F[N_X-3]))**2) *\
                (0.5*inv_dy*(3.0 * T[N_Y-1, N_X-1] - 4.0 * T[N_Y-2, N_X-1] + T[N_Y-3, N_X-1]))/F[N_X-1]

    return F_new
