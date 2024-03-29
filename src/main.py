from two_phase.nonuniform_y_grid.schemes.ADI import solve
from two_phase.nonuniform_y_grid.boundary import init_f_vector, recalculate_boundary
from two_phase.nonuniform_y_grid.temperature import init_temperature, solar_heat, air_temperature
from two_phase.nonuniform_y_grid.plotting import plot_non_transformed
from two_phase.nonuniform_y_grid.grid_generation import get_node_coord
from parameters import *
import numpy as np
import time


# ВХОДНАЯ ТОЧКА ПРОГРАММЫ
if __name__ == '__main__':

    # Инициализируем положение границы фазового перехода в начальный момент времени
    # в НОВЫХ координатах
    F = init_f_vector(n_x=N_X)

    # Начальное распределение температуры в НОВЫХ координатах
    T = init_temperature(F)

    # T = np.load("data/analytical.npz")['T']

    path = f"graphs/temperature/H={H}_NX={N_X}_NY={N_Y}_dt={round(dt*t_0)}_conv={round(conv_coef)}"

    # График начального распределения температуры (в исходных координатах)
    # plot_non_transformed(
    #     T=T,
    #     F=F,
    #     time=0,
    #     graph_id=0,
    #     path=path
    # )

    np.savez_compressed("data/f_and_temp_at_0", F=F, T=T)

    # Инициализируем переменные для температуры и положения свободной границы на новом шаге по времени
    T_new = np.copy(T)
    T_old = np.copy(T)
    F_new = np.copy(F)
    F_old = np.copy(F)

    j_int = int(0.5 * (N_Y - 1))  # координата границы фазового перехода в новых координатах

    # Инициализируем неравномерную сетку по координате Y
    Y = np.empty(N_Y)
    for j in range(N_Y):
        Y[j] = get_node_coord(j, j_int)
        # print(Y[j])

    t_step = 1  # Номер шага по времени
    K = 2  # Число итераций на одном шаге

    start_time = time.process_time()  # Начальное время расчетов

    result = []

    while t_step < N_t:
        # Итерационный метод
        for k in range(K):
            T_new = solve(
                T=T_old,
                F_new=F_new,
                F_old=F_old,
                Y=Y,
                time=t_step * dt * t_0
            )
            F_new = recalculate_boundary(
                F=F_old,
                T=T_new
            )

        if np.amax(F_new) >= H or np.amin(F_new) <= 0:
            print("### ФАЗОВЫЙ ПЕРЕХОД ДОШЕЛ ДО ГРАНИЦЫ ОБЛАСТИ ###")
            print(f"ШАГ: {t_step}")
            break

        T_old = np.copy(T_new)
        F_old = np.copy(F_new)

        # print("### ТЕМПЕРАТУРА НА НОВОМ ШАГЕ РАССЧИТАНА ###")
        if t_step % 3600 == 0:
            print(f"### ВРЕМЯ ВЫПОЛНЕНИЯ: {time.process_time() - start_time} ###")
            print(f"ШАГ: {t_step}")
            model_time = round(t_step * dt * t_0 / 3600.0, 2)

            # if t_step * 5 / 3600 < 26:
            #     print(f"T_air = {air_temperature(t_step * dt * t_0)}, Q_sol = {solar_heat(t_step * dt * t_0)}")
            #     print("### СОХРАНЯЮ ГРАФИК ###")
            #     plot_non_transformed(
            #         T=T_new,
            #         F=F_new,
            #         time=model_time,
            #         graph_id=t_step,
            #         path=path
            #     )

            print(f"### СОХРАНЯЮ ПОЛОЖЕНИЕ ГРАНИЦЫ И ТЕМПЕРАТУРНОЕ РАСПРЕДЕЛЕНИЕ В АРХИВ"
                  f" data/f_and_temp_at_{t_step}.npz ###")
            np.savez_compressed(f"data/f_and_temp_at_{t_step}", F=F_new, T=T_new)

        t_step = t_step + 1

    print("### РАСЧЁТ ЗАВЕРШЁН ###")
