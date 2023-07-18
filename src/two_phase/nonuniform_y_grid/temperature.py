import numpy as np
import math
import numba
from parameters import *
from two_phase.nonuniform_y_grid.grid_generation import get_node_coord


@numba.jit
def air_temperature(t: float):
    """
    Функция изменения времени на поверхности воды.
    :param t: Время в секундах
    :return: Температура на поверхности в заданный момент времени
    """
    # return T_air
    return T_air + T_amp * math.sin(2 * math.pi * t / (24.0 * 3600.0) - math.pi / 2)


def init_temperature(F):
    """
    Задает начальное распределение температуры на сетке в НОВЫХ координатах.
    :return: двумерный массив со значениями температуры в начальный момент времени
    """
    T = np.empty((N_Y, N_X))

    j_int = int(0.5*(N_Y - 1))  # координата границы фазового перехода в новых координатах

    f_max = np.amax(F)
    f_min = np.amin(F)

    for i in range(N_X):
        for j in range(N_Y):
            if j < j_int:
                # T[j, :] = T_ice / T_0  # задаем температуру льда

                # if F[i] * get_node_coord(j, j_int) < f_min:
                #     T[j, i] = T_ice / T_0
                # else:
                #     T[j, i] = T_ice / T_0 + (F[i] * get_node_coord(j, j_int) - f_min) * (1.0 - T_ice / T_0) / (f_max - f_min)

                T[j, i] = T_ice / T_0 + F[i] * get_node_coord(j, j_int) * (1.0 - T_ice / T_0) / f_max  # задаем температуру льда
            else:
                T[j, i] = T_w / T_0  # задаем температуру воды
                # T[j, i] = 1.0 + get_node_coord(j, j_int) * (T_w / T_0 - 1.0) / 2.0  # задаем температуру льда
    T[j_int, :] = T_0/T_0
    T[N_Y - 1, :] = air_temperature(0.0)/T_0  # температура воды на "верхней" границе равна температуре воздуха

    return T

