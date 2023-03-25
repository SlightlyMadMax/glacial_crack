#
# ФИЗИЧЕСКИЕ КОНСТАНТЫ
#

# Удельная теплоемкость воды при 0°C [J/(kg*K)]
c_w = 4217.0

# Плотность воды при 0 °C [kg/m^3]
rho_w = 999.84

# Коэффициент теплопроводности воды при 0°C [W/(m*K)]
k_w = 0.569

# Удельная теплоемкость льда при -10°C [J/(kg*K)]
c_ice = 2000.0

# Плотность льда при -10°C [kg/m^3]
rho_ice = 918.9

# Коэффициент теплопроводности льда при -10°C [W/(m*K)]
k_ice = 2.3

# Удельная теплота плавления льда [J/kg]
L = 334000.0


#
# ПАРАМЕТРЫ ДЛЯ ЗАДАНИЯ НАЧАЛЬНОГО ПОЛОЖЕНИЯ ГРАНИЦЫ ФАЗОВОГО ПЕРЕХОДА [М]
#

h = .2

#
# ПАРАМЕТРЫ ОБЛАСТИ [М]
#

# Высота
H = 1.2
# Ширина
W = 1.0

#
# ТЕМПЕРАТУРА
#

# НАЧАЛЬНАЯ ТЕМПЕРАТУРА ЛЬДА
T_ice = 263.15  # -10°C
# ТЕМПЕРАТУРА ВОЗДУХА (пока не используется)
T_air = 278.15  # 5°C
# ТЕМПЕРАТУРА ФАЗОВОГО ПЕРЕХОДА ВОДА-ЛЁД
T_0 = 273.15  # 0°C


#
# МЕТОД КОНЧЕНЫХ РАЗНОСТЕЙ
#

N_X = 400  # Число узлов сетки по оси X
N_Y = 400  # Число узлов сетки по оси Y

# Шаги на сетке
dx = 1.0 / (N_X - 1)  # x в НОВЫХ координатах меняется от 0 до 1
dy = 1.0 / (N_Y - 1)  # y в НОВЫХ координатах меняется от 0 до 1

t_0 = (c_ice*rho_ice*W*W)/k_ice  # Параметр обезразмеривания времени
end_time = 3600.0*48/t_0  # Конечное время
dt = 60.0/t_0  # Шаг по времени

N_t = int(end_time/dt)  # Число шагов по времени


gamma = L/(2050.0*W*W*T_0)  # Параметр, используемый в условии Стефана в новых координатах.
