import math
from parameters import s


def get_node_coord(t: float):
    """
    Функция для генерации неравномерной сетки. Сгущает узлы ближе к фазовой границе.
    :param t: координата на равномерной сетке на интервале (0, 2).
    :return: координата на неравномерной сетке.
    """
    return 1.0 - math.log(2.0 / t - 1.0) / s
