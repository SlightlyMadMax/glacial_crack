from one_phase.temperature import *
from one_phase.fd_scheme import predict_correct
import numpy as np
from parameters import *
from plotting import plot_temperature


# ВХОДНАЯ ТОЧКА ПРОГРАММЫ
if __name__ == '__main__':

    # Сетка в преобразованных координатах (по x пока не масштабируем для простоты)
    x = np.linspace(0, W, N_X, endpoint=True)
    y = np.linspace(0, 1, N_Y, endpoint=True)

    # Шаги на сетке
    dx = W/(N_X - 1)  # x, как в новых, так и в старых координатах меняется от 0 до W
    dy = 1/(N_Y - 1)  # y в НОВЫХ координатах меняется от 0 до 1

    # Инициализируем положение границы фазового перехода в начальный момент времени
    F = init_f_vector(x=x)

    # Начальное распределение температуры в НОВЫХ координатах
    T = init_temperature()

    # График начального распределения температуры (в исходных координатах)
    plot_temperature(
        T=reverse_transform(T, F),  # Преобразуем к исходным координатам
        time=0,
        graph_id=0
    )

    t_step = 1  # Номер шага по времени

    # Инициализируем переменные для температуры и положения свободной границе на новом шаге по времени
    T_new = np.copy(T)
    F_new = np.copy(F)

    while t_step < N_t:
        print("### ВЫЧИСЛЯЮ ПОЛОЖЕНИЕ ГРАНИЦЫ, ШАГ = " + str(t_step) + " ###")
        F_new = recalculate_boundary(F=F_new, T=T, dy=dy)
        print("### ВЫЧИСЛЯЮ ТЕМПЕРАТУРУ, ШАГ = " + str(t_step) + " ###")
        T_new = predict_correct(
            T=T_new,
            F_new=F_new,
            F_old=F,
            dx=dx,
            dy=dy
        )
        print("### ТЕМПЕРАТУРА НА НОВОМ ШАГЕ РАССЧИТАНА ###")
        print("### СОХРАНЯЮ ГРАФИК ###")
        plot_temperature(
            T=reverse_transform(T_new, F_new),  # Преобразуем к исходным координатам
            time=round(t_step*(dt*t_0), 2),
            graph_id=t_step
        )
        F = np.copy(F_new)  # В F храним положение границы на предыдущем шаге по времени
        t_step = t_step + 1
    print("### РАСЧЁТ ЗАВЕРШЁН ###")
