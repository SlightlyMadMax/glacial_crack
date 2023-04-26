import numpy as np
from parameters import *


def init_temperature():
    """
    Задание начального распределения температуры на сетке в НОВЫХ координатах.
    """
    T = np.empty((N_Y, N_X))

    j_int = int(0.5*(N_Y - 1))  # координата границы фазового перехода в новых координатах

    for j in range(N_Y):
        if j < j_int:
            T[j, :] = T_ice/T_0  # задаем температуру льда
        else:
            T[j, :] = T_w/T_0  # задаем температуру воды
    print(j_int)
    T[j_int, :] = T_0/T_0
    T[N_Y - 1, :] = T_air/T_0  # температура воды на "верхней" границе равна температуре воздуха
    print(T)
    return T

