from parameters import W, H, N_X, N_Y, dt, t_0, T_0
import matplotlib.pyplot as plt
import numpy as np


def plot_non_transformed(T, F, time: float, graph_id: int):
    """
    Строит график по двумерному массиву температуры, заданному в НОВЫХ координатах.
    Осуществляет перевод в исходные координаты, построение и сохранение графика.
    :param T: двумерный массив температуры
    :param F: вектор с координатами границы фазового перехода
    :param time: время в часах
    :param graph_id: идентификатор или порядковый номер графика
    :return: None
    """
    x = np.linspace(0, 1.0, N_X)
    y = np.linspace(0, 2.0, N_Y)
    j_int = int(0.5 * (N_Y - 1))

    X, Y = np.meshgrid(x, y)

    X = X * W
    for i in range(N_X):
        for j in range(N_Y):
            if j <= j_int:
                Y[j, i] = Y[j, i] * F[i]
            else:
                Y[j, i] = (Y[j, i] - 1.0) * (H - F[i]) + F[i]

    fig = plt.figure()
    ax = plt.axes()
    # plt.plot(X, Y, marker=".", markersize=0.5, color='red', linestyle='none')  # сетка
    plt.contourf(X, Y, T_0 * T - T_0, 50, cmap="viridis")
    plt.colorbar()
    ax.set_title(f"time = {str(time)} h\n dx = 1/{str(N_X)} m, dy = 1/{str(N_Y)} m, dt = {str(round(dt * t_0 / 3600.0, 2))} h")
    ax.set_xlabel("x, m")
    ax.set_ylabel("y, m")
    plt.savefig(f"graphs/temperature/T_{str(graph_id)}.png")
    plt.show()
    # plt.close()
