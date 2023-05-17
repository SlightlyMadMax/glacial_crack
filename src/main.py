from two_phase.nonuniform_y_grid.schemes.ADI import solve
from two_phase.nonuniform_y_grid.boundary import init_f_vector, recalculate_boundary
from two_phase.nonuniform_y_grid.temperature import init_temperature
from two_phase.nonuniform_y_grid.plotting import plot_non_transformed
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
            print("### ФАЗОВЫЙ ПЕРЕХОД ДОШЕЛ ДО ВЕРХНЕЙ ГРАНИЦЫ ОБЛАСТИ ###")
            break

        T_old = np.copy(T_new)
        F_old = np.copy(F_new)

        # print("### ТЕМПЕРАТУРА НА НОВОМ ШАГЕ РАССЧИТАНА ###")
        if t_step % 180 == 0:
            print(f"### ELAPSED CPU TIME: {time.process_time() - start_time} ###")

            model_time = round(t_step * dt * t_0 / 3600.0, 2)

            print("### СОХРАНЯЮ ГРАФИК ###")
            plot_non_transformed(
                T=T_new,
                F=F_new,
                time=model_time,
                graph_id=t_step
            )

            print(f"### СОХРАНЯЮ ПОЛОЖЕНИЕ ГРАНИЦЫ И ТЕМПЕРАТУРНОЕ РАСПРЕДЕЛЕНИЕ В АРХИВ"
                  f" data/f_and_temp_at_{t_step}.npz ###")
            np.savez_compressed(f"data/f_and_temp_at_{t_step}", F=F_new, T=T_new)

        t_step = t_step + 1

    print("### РАСЧЁТ ЗАВЕРШЁН ###")
