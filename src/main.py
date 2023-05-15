from two_phase.non_uniform_y_grid.schemes.ADI import solve
from two_phase.non_uniform_y_grid.boundary import init_f_vector, recalculate_boundary
from two_phase.non_uniform_y_grid.temperature import init_temperature
from parameters import *
from two_phase.non_uniform_y_grid.plotting import plot_non_transformed
import numpy as np
import time


# ВХОДНАЯ ТОЧКА ПРОГРАММЫ
if __name__ == '__main__':

    # Инициализируем положение границы фазового перехода в начальный момент времени
    # в НОВЫХ координатах
    F = init_f_vector(n_x=N_X)

    # Начальное распределение температуры в НОВЫХ координатах
    T = init_temperature(F)

    # График начального распределения температуры (в исходных координатах)
    plot_non_transformed(
        T=T,
        F=F,
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

    start_time = time.process_time()

    while t_step < N_t:
        # print("### ВЫЧИСЛЯЮ ПОЛОЖЕНИЕ ГРАНИЦЫ, ШАГ = " + str(t_step) + " ###")
        # print("### ВЫЧИСЛЯЮ ТЕМПЕРАТУРУ, ШАГ = " + str(t_step) + " ###")

        # Итерационный метод
        for k in range(K):
            T_new = solve(
                T=T_old,
                F_new=F_new,
                F_old=F_old,
                time=t_step * dt * t_0
            )
            F_new = recalculate_boundary(F=F_old, T=T_new)

        if np.amax(F_new) >= H:
            print("Фазовый переход дошел до верхней границы области.")
            break

        T_old = np.copy(T_new)
        F_old = np.copy(F_new)

        # print("### ТЕМПЕРАТУРА НА НОВОМ ШАГЕ РАССЧИТАНА ###")
        # print("### СОХРАНЯЮ ГРАФИК ###")
        if t_step % 1 == 0:
            print(f"Elapsed CPU time: {time.process_time() - start_time}")
            plot_non_transformed(
                T=T_new,
                F=F_new,
                time=t_step * dt * t_0 / 3600.0,
                graph_id=t_step
            )
            # print(f"Days: {t_step/20}")
            # result.append(F_new[15])
            # print(F_new[15])
        t_step = t_step + 1

    print("### РАСЧЁТ ЗАВЕРШЁН ###")
    print(result)
