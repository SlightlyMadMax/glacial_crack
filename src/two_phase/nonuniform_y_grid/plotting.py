from parameters import W, H, N_X, N_Y, dt, t_0, T_0, conv_coef, k_w
import matplotlib.pyplot as plt
import numpy as np
from two_phase.nonuniform_y_grid.grid_generation import get_node_coord
from two_phase.nonuniform_y_grid.temperature import air_temperature


def plot_non_transformed(T, F, time: float, graph_id: int, non_uniform: bool = True):
    """
    Строит график по двумерному массиву температуры, заданному в НОВЫХ координатах.
    Осуществляет перевод в исходные координаты, построение и сохранение графика.
    :param T: двумерный массив температуры
    :param F: вектор с координатами границы фазового перехода
    :param time: время в часах
    :param graph_id: идентификатор или порядковый номер графика
    :param non_uniform: температура задана на однородной или неоднородной сетке
    :return: None
    """
    x = np.linspace(0, 1.0, N_X)
    y = np.empty(N_Y)
    j_int = int(0.5 * (N_Y - 1))

    for j in range(N_Y):
        y[j] = get_node_coord(j, j_int)

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

    # ax.set_aspect("equal")
    # plt.plot(X, Y, marker=".", markersize=0.5, color='k', linestyle='none')  # сетка
    plt.plot(X[j_int, :], Y[j_int, :], linewidth=1, color='r', label='Граница ф.п.')  # граница ф.п.
    # plt.legend(loc="upper right")
    plt.contourf(X, Y, T_0*T - T_0, 100, cmap="viridis")

    # plt.ylim((9.9, 10.05))
    # plt.xlim((0.3, 0.7))


    plt.colorbar()
    # plt.clim(-5, 5)

    # if non_uniform:
    #     title = f"time = {time} h, T_air = {round(air_temperature(graph_id * dt * t_0) - 273.15, 2)} C, " \
    #             f"non-uniform grid\nN_X = {N_X}, N_Y = {N_Y}, s = {s}, dt = {round(dt * t_0 / 3600.0, 2)} h"
    # else:
    #     title = f"time = {time} h\n dx = 1/{N_X} m, dy = 1/{N_Y} m, dt = {round(dt * t_0 / 3600.0, 2)} h"
    #

    title = f"Time = {time} h, T_air = {round(air_temperature(graph_id * dt * t_0) - 273.15, 2)} C,\n" \
            f"Heat transfer coef = {round(conv_coef*k_w)} W / (K * m^2)"
    ax.set_title(title)
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")

    # plt.savefig(f"graphs/temperature/T_{graph_id}.eps", format="eps")  # сохранить в векторном формате
    plt.savefig(f"graphs/temperature/T_{graph_id}.png")  # сохранить в растровом формате

    # ax.set_aspect("equal")
    #
    # plt.ylim((9.90, 10.05))
    # plt.xlim((0.3, 0.7))
    #
    # plt.savefig(f"graphs/temperature/above_water_{graph_id}.png")
    #
    # plt.ylim((9.90, 10.05))
    # plt.xlim((0.3, 0.45))
    #
    # plt.savefig(f"graphs/temperature/left_water_{graph_id}.png")

    plt.show()
    # plt.close()
