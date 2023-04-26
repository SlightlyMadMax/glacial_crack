from parameters import s
import math


def get_node_coord(t: float):
    """
    Функция для генерации неравномерной сетки. Сгущает узлы ближе к единице.
    :param t: координата на равномерной сетке на интервале (0, 1).
    :return: координата на неравномерной сетке.
    """
    return 1.0 - math.exp(-s * t) + math.exp(-s)
