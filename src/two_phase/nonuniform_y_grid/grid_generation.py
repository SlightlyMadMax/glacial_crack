import math
import numba
from parameters import s_ice, s_w, N_Y


@numba.jit
def get_node_coord(j: int, j_int: int) -> float:
    """
    Функция для генерации неравномерной сетки. Сгущает узлы ближе к фазовой границе.
    :param t: координата на равномерной сетке на интервале (0, 1).
    :return: координата на неравномерной сетке.
    """
    if j == 0:
        return 0.0
    elif j == j_int:
        return 1.0
    elif j == N_Y - 1:
        return 2.0
    elif j < j_int:
        t = 2 * j / (N_Y - 2)
        return 1.0 - math.exp(-s_ice * t) + math.exp(-s_ice)
    else:
        t = (2 * N_Y - 4 - 2 * j) / (N_Y - 2)
        return 1.0 + math.exp(-s_w * t) - math.exp(-s_w)
