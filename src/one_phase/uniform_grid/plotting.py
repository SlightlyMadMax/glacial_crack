from parameters import W, N_X, N_Y, dt, t_0, T_0
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
    y = np.linspace(0, 1.0, N_Y)

    X, Y = np.meshgrid(x, y)

    X = X * W
    Y[:, :] = Y[:, :] * F[:]

    fig = plt.figure()
    ax = plt.axes()
    # plt.plot(X, Y[j_int, :], marker=".", markersize=0.1, color='red', linestyle='none')  # сетка
    plt.contourf(X, Y, T_0 * T - T_0, 50, cmap="viridis")
    plt.colorbar()

    ax.set_title(f"time = {time} h\n dx = 1/{N_X} m, dy = 1/{N_Y} m, dt = {round(dt * t_0 / 3600.0, 2)} h")
    ax.set_xlabel("x, m")
    ax.set_ylabel("y, m")
    plt.savefig(f"graphs/temperature/T_{graph_id}.png")
    plt.show()
    # plt.close()
