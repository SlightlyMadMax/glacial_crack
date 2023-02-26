from parameters import W, H


def plot_temperature(T, time: float, graph_id: int):
    """
    Построение графика
    T – матрица со значениями температуры на двумерной сетке в исходных координатах
    time – время
    graph_id – id графика
    """

    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    fig = plt.figure(figsize=(4, 8))
    ax = plt.axes()
    plt.imshow(T
               , extent=[0, W, 0, H]
               , origin='lower'
               , cmap='winter'
               , interpolation='none'
               , vmin=-20
               , vmax=0
               )
    plt.colorbar()
    ax.set_title('time = ' + str(time) + 's')
    ax.set_xlabel('x, m')
    ax.set_ylabel('y, m')
    plt.savefig('../graphs/T_' + str(graph_id) + '.png')
    plt.show()


