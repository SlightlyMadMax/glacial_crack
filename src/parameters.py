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
c_ice = 2097.0

# Плотность льда при -10°C [kg/m^3]
rho_ice = 918.9

# Плотность льда при 0°C [kg/m^3]
rho_ice_0 = 916.2

# Коэффициент теплопроводности льда при -10°C [W/(m*K)]
k_ice = 2.1

# Удельная теплота плавления льда [J/kg]
L = 334000.0


#
# ПАРАМЕТРЫ ДЛЯ ЗАДАНИЯ НАЧАЛЬНОГО ПОЛОЖЕНИЯ ГРАНИЦЫ ФАЗОВОГО ПЕРЕХОДА [М]
#

a = 10.0

#
# ПАРАМЕТРЫ ОБЛАСТИ [М]
#

# Высота
H = 10.05
# Ширина
W = 1.0

#
# ТЕМПЕРАТУРА
#

# ТЕМПЕРАТУРА ФАЗОВОГО ПЕРЕХОДА ВОДА-ЛЁД
T_0 = 273.15  # 0°C
# НАЧАЛЬНАЯ ТЕМПЕРАТУРА ЛЬДА
T_ice = 253.15  # -20°C
# НАЧАЛЬНАЯ ТЕМПЕРАТУРА ВОДЫ
T_w = 273.15  # 0°C
# НАЧАЛЬНАЯ ТЕМПЕРАТУРА ВОЗДУХА
T_air = 275.15  # 2°C
# АМПЛИТУДА ИЗМЕНЕНИЯ ТЕМПЕРАТУРЫ ВОЗДУХА
T_amp = 2.0


#
# МЕТОД КОНЧЕНЫХ РАЗНОСТЕЙ
#

N_X = 1000  # Число узлов сетки по оси X
N_Y = 200  # Число узлов сетки по оси Y

# Шаги на сетке
dx = 1.0 / (N_X - 1)  # x в НОВЫХ координатах меняется от 0 до 1
dy = 1.0 / (N_Y - 1)  # y в НОВЫХ координатах меняется от 0 до 1

s_ice = 8.0  # Параметр, отвечающий за густоту сетки у границы, чем больше – тем гуще
s_w = 4.5


t_0 = (c_ice*rho_ice*W*W)/k_ice  # Параметр обезразмеривания времени
end_time = (3600.0*24.0*7.0)/t_0  # Конечное время
dt = 1.0/t_0  # Шаг по времени

N_t = int(end_time/dt)  # Число шагов по времени


gamma = (L*rho_ice_0)/(c_ice*rho_ice*W*W*T_0)  # Параметр, используемый в условии Стефана в новых координатах.

conv_coef = 600.0/k_w
