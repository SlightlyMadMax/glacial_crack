from parameters import W, H, N_X, N_Y, dt, t_0


def plot_temperature(T, time: float, graph_id: int):
    """
    Построение графика
    T – матрица со значениями температуры на двумерной сетке в исходных координатах
    time – время
    graph_id – id графика
    """

    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes()
    plt.imshow(T
               , extent=[0, W, 0, H]
               , origin='lower'
               , cmap='winter'
               , interpolation='none'
               , vmin=-10
               , vmax=0
               )
    plt.colorbar()
    ax.set_title(
        'time = ' + str(time) + ' h\n' +
        'dx = 1/' + str(N_X) + ' m, dy = 1/' + str(N_Y) +
        ' m, dt = ' + str(round(dt * t_0 / 3600.0, 2)) + ' h'
    )
    ax.set_xlabel('x, m')
    ax.set_ylabel('y, m')
    plt.savefig('../graphs/T_' + str(graph_id) + '.png')
    # plt.show()


