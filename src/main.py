from one_phase.nonuniform_grid.schemes.ADI import solve
from one_phase.nonuniform_grid.boundary import init_f_vector, recalculate_boundary
from one_phase.nonuniform_grid.temperature import reverse_transform, init_temperature
from parameters import *
from plotting import plot_temperature
import numpy as np
import time


# ВХОДНАЯ ТОЧКА ПРОГРАММЫ
if __name__ == '__main__':

    # Инициализируем положение границы фазового перехода в начальный момент времени
    # в НОВЫХ координатах
    F = init_f_vector(n_x=int(N_X))

    # Начальное распределение температуры в НОВЫХ координатах
    T = init_temperature()

    # График начального распределения температуры (в исходных координатах)
    plot_temperature(
        T=reverse_transform(T, F),  # Преобразуем к исходным координатам
        time=0,
        graph_id=0
    )

    t_step = 1  # Номер шага по времени

    # Инициализируем переменные для температуры и положения свободной границы на новом шаге по времени
    T_new = np.copy(T)
    T_old = np.copy(T)
    F_new = np.copy(F)
    F_old = np.copy(F)

    K = 2  # Число итераций на одном шаге
    result = []
    while t_step < N_t:
        # print("### ВЫЧИСЛЯЮ ПОЛОЖЕНИЕ ГРАНИЦЫ, ШАГ = " + str(t_step) + " ###")
        # print("### ВЫЧИСЛЯЮ ТЕМПЕРАТУРУ, ШАГ = " + str(t_step) + " ###")

        # print("T_old")
        # print(T_old)
        # print("F_old")
        # print(F_old)
        # Итерационный метод
        for k in range(K):
            T_new = solve(
                T=T_old,
                F_new=F_new,
                F_old=F_old,
            )
            F_new = recalculate_boundary(F=F_old, T=T_new)

        T_old = np.copy(T_new)
        F_old = np.copy(F_new)
        # print("T_new")
        # print(T_new)
        # print("F_new")
        #print(F_new)
        # print("### ТЕМПЕРАТУРА НА НОВОМ ШАГЕ РАССЧИТАНА ###")
        # print("### СОХРАНЯЮ ГРАФИК ###")
        if t_step % 20 == 0:
            plot_temperature(
                T=reverse_transform(T_new, F_new),  # Преобразуем к исходным координатам
                time=round(t_step * (dt * t_0/3600.0), 1),
                graph_id=t_step
            )
            # result.append(F_new[25])
            # print(F_new[25])
        t_step = t_step + 1

    print("### РАСЧЁТ ЗАВЕРШЁН ###")
    print(result)
