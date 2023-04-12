import numpy as np
from parameters import *
from src.one_phase.nonuniform_y_grid.schemes.ADI import sigmoid


def init_f_vector(n_x):
    """
    Задание начального положения границы фазового перехода.
    """
    F = np.empty(n_x)
    for i in range(n_x):
        # F[i] = a
        # F[i] = a + 2*(i*dx - W/2)*(i*dx - W/2)  # Парабола f(x, t=0) = 3*(x - W/2)^2 + h
        # if i*dx < 0.2 or i*dx > 0.8:
        #     F[i] = h
        # else:
        #     F[i] = 1.0 - 1.5*i*dx if i*dx < W/2 else -0.5 + 1.5*i*dx

        if i*dx < 0.4 or i*dx > 0.6:
            F[i] = 1.0
        else:
            F[i] = 3.8 - 7.0*i*dx if i*dx < 0.5 else 7.0*i*dx - 3.2
    return F


def recalculate_boundary(F, T):
    """
    Пересчёт положения границы фазового перехода в соответствии с условием Стефана в новых координатах.
    F – вектор значений положения границы фазового перехода.
    T – матрица значений температуры на сетке в НОВЫХ координатах.
    dx, dy – шаги по x и y на сетке в НОВЫХ координатах.
    """
    F_new = np.copy(F)
    inv_dx = 1.0/dx
    inv_W = 1.0/W
    inv_gamma = 1.0/gamma

    h = 1.0 - sigmoid((N_Y - 2) / (N_Y - 1))
    h_0 = sigmoid((N_Y - 2) / (N_Y - 1)) - sigmoid((N_Y - 3) / (N_Y - 1))

    for i in range(1, N_X-1):
        F_new[i] = F[i] + dt*inv_gamma*(1.0 + (0.5*inv_W*inv_dx*(F[i+1]-F[i-1]))*(0.5*inv_W*inv_dx*(F[i+1]-F[i-1]))) * \
                   ((T[N_Y-1, i] - T[N_Y-2, i])*(h + h_0)*(h + h_0) + (T[N_Y-3, i]-T[N_Y-1, i])*h*h)/(h_0*h*(h_0+h)*F[i])

    F_new[0] = F[0] + dt*inv_gamma*(1.0 + (0.5*inv_W*inv_dx*(4.0*F[1]-3.0*F[0]-F[2]))*(0.5*inv_W*inv_dx*(4.0*F[1]-3.0*F[0]-F[2]))) *\
               ((T[N_Y-1, 0] - T[N_Y-2, 0])*(h + h_0)*(h + h_0) + (T[N_Y-3, 0]-T[N_Y-1, 0])*h*h)/(h_0*h*(h_0+h)*F[0])

    F_new[N_X-1] = F[N_X-1] + dt*inv_gamma*(1.0 + (0.5*inv_W*inv_dx*(3.0*F[N_X-1]-4.0*F[N_X-2]+F[N_X-3]))**2) *\
                ((T[N_Y-1, N_X-1] - T[N_Y-2, N_X-1])*(h + h_0)*(h + h_0) + (T[N_Y-3, N_X-1]-T[N_Y-1, N_X-1])*h*h)/(h_0*h*(h_0+h)*F[N_X-1])

    return F_new
