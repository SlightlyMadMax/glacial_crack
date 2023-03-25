import numpy as np
from parameters import *


def init_f_vector(n_x):
    """
    Задание начального положения границы фазового перехода.
    """
    F = np.empty(n_x)
    for i in range(n_x):
        # F[i] = h
        # F[i] = h + 2*(i*dx - W/2)*(i*dx - W/2)  # Парабола f(x, t=0) = 3*(x - W/2)^2 + h
        # if i*dx < 0.2 or i*dx > 0.8:
        #     F[i] = h
        # else:
        #     F[i] = 1.0 - 1.5*i*dx if i*dx < W/2 else -0.5 + 1.5*i*dx
        #
        if i*dx < 0.4 or i*dx > 0.6:
            F[i] = 1.0
        else:
            F[i] = 3.8 - 7.0*i*dx if i*dx < 0.5 else 7.0*i*dx - 3.2
    return F


def init_temperature():
    """
    Задание начального распределения температуры на сетке в НОВЫХ координатах.
    """
    T = np.empty((N_Y, N_X))
    T[:, :] = T_ice/T_0  # Температура льда
    T[N_Y-1, :] = 1.0  # Температура фазового перехода
    return T


def recalculate_boundary(F, T):
    """
    Пересчёт положения границы фазового перехода в соответствии с условием Стефана в новых координатах.
    F – вектор значений положения границы фазового перехода.
    T – матрица значений температуры на сетке в НОВЫХ координатах.
    dx, dy – шаги по x и y на сетке в НОВЫХ координатах.
    """
    F_new = np.copy(F)
    inv_dy = 1.0/dy
    inv_dx = 1.0/dx
    inv_W = 1.0/W
    inv_gamma = 1.0/gamma

    for i in range(1, N_X-1):
        inv_F = 1.0/F[i]
        F_new[i] = F[i] + dt*inv_gamma*inv_F*(1.0 + (0.5*inv_W*inv_dx*(F[i+1]-F[i-1]))**2) *\
                   (0.5*inv_dy*(3.0 * T[N_Y - 1, i] - 4.0 * T[N_Y - 2, i] + T[N_Y - 3, i]))
    F_new[0] = F[0] + dt*inv_gamma*(1.0 + (0.5*inv_W*inv_dx*(4.0*F[1]-3.0*F[0]-F[2]))**2) *\
               (0.5*inv_dy*(3.0 * T[N_Y-1, 0] - 4.0 * T[N_Y-2, 0] + T[N_Y-3, 0]))/F[0]
    F_new[N_X-1] = F[N_X-1] + dt*inv_gamma*(1.0 + (0.5*inv_W*inv_dx*(3.0*F[N_X-1]-4.0*F[N_X-2]+F[N_X-3]))**2) *\
                (0.5*inv_dy*(3.0 * T[N_Y-1, N_X-1] - 4.0 * T[N_Y-2, N_X-1] + T[N_Y-3, N_X-1]))/F[N_X-1]
    return F_new


def reverse_transform(T, F):
    """
    Пересчёт температуры в исходные координаты.
    T – матрица значений температуры на сетке в НОВЫХ координатах.
    F – вектор значений положения границы фазового перехода.
    """
    new_H = int(H*N_Y)
    new_W = int(W*N_X)

    T_new = np.zeros((new_H, new_W))

    for i in range(N_X):
        new_x = int(round(i*W))
        for j in range(N_Y):
            new_y = int(round(j * F[i]))
            if new_y < new_H and new_x < int(W*N_X):
                T_new[new_y, new_x] = T_0*T[j, i] - T_0  # Пересчитываем в ИСХОДНЫЕ координаты и переходим к °С
    return T_new

