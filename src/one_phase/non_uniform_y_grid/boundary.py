import numpy as np
from parameters import *
from src.one_phase.non_uniform_y_grid.grid_generation import get_node_coord


def init_f_vector(n_x):
    """
    Задание начального положения границы фазового перехода.
    :param n_x: число узлов по x
    :return: вектор с координатами границы фазового перехода
    """
    F = np.empty(n_x)
    for i in range(n_x):
        # Плоскость
        # F[i] = a

        # Парабола f(x, t=0) = 3*(x - W/2)^2 + h
        # F[i] = a + 2*(i*dx - W/2)*(i*dx - W/2)

        # Подобие трещины
        if i*dx < 0.4 or i*dx > 0.6:
            F[i] = 1.0
        else:
            F[i] = 3.8 - 7.0*i*dx if i*dx < 0.5 else 7.0*i*dx - 3.2

    return F


def recalculate_boundary(F, T):
    """
    Пересчёт положения границы фазового перехода в соответствии с условием Стефана в новых координатах.
    :param F: вектор значений положения границы фазового перехода
    :param T: матрица значений температуры на сетке в НОВЫХ координатах
    :return: вектор с координатами границы фазового перехода
    """
    F_new = np.copy(F)
    inv_dx = 1.0 / dx
    inv_W = 1.0 / W
    inv_gamma = 1.0 / gamma

    h = 1.0 - get_node_coord((N_Y - 2) / (N_Y - 1))
    h_0 = get_node_coord((N_Y - 2) / (N_Y - 1)) - get_node_coord((N_Y - 3) / (N_Y - 1))

    for i in range(1, N_X-1):
        df_dx = 0.5 * inv_W * inv_dx * (F[i+1]-F[i-1])
        dT_dy = ((T[N_Y-1, i] - T[N_Y-2, i])*(h + h_0)*(h + h_0) + (T[N_Y-3, i]-T[N_Y-1, i])*h*h)/(h_0*h*(h_0+h))
        F_new[i] = F[i] + dt * inv_gamma * (1.0 + df_dx * df_dx) * dT_dy / F[i]

    df_dx_0 = 0.5 * inv_W * inv_dx * (4.0*F[1]-3.0*F[0]-F[2])
    dT_dy_0 = ((T[N_Y-1, 0] - T[N_Y-2, 0])*(h + h_0)*(h + h_0) + (T[N_Y-3, 0]-T[N_Y-1, 0])*h*h)/(h_0*h*(h_0+h))
    F_new[0] = F[0] + dt * inv_gamma * (1.0 + df_dx_0 * df_dx_0) * dT_dy_0 / F[0]

    df_dx_n = 0.5 * inv_W * inv_dx * (3.0*F[N_X-1]-4.0*F[N_X-2]+F[N_X-3])
    dT_dy_n = ((T[N_Y-1, N_X-1]-T[N_Y-2, N_X-1])*(h+h_0)*(h+h_0)+(T[N_Y-3, N_X-1]-T[N_Y-1, N_X-1])*h*h)/(h_0*h*(h_0+h))
    F_new[N_X-1] = F[N_X-1] + dt * inv_gamma * (1.0 + df_dx_n * df_dx_n) * dT_dy_n / F[N_X-1]

    return F_new
