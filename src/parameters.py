#
# ФИЗИЧЕСКИЕ КОНСТАНТЫ
#

# Удельная теплоемкость воды при 0°C [J/(kg*K)]
c_w = 4217

# Плотность воды при 0 °C [kg/m^3]
rho_w = 999.84

# Коэффициент теплопроводности воды при 0°C [W/(m*K)]
k_w = 0.569

# Удельная теплоемкость льда при -20°C [J/(kg*K)]
c_ice = 1943

# Плотность льда при -20°C [kg/m^3]
rho_ice = 919.4

# Коэффициент теплопроводности льда при -20°C [W/(m*K)]
k_ice = 2.39

# Удельная теплота плавления льда [J/kg]
L = 334000


#
# ПАРАМЕТРЫ ДЛЯ ЗАДАНИЯ НАЧАЛЬНОГО ПОЛОЖЕНИЯ ГРАНИЦЫ ФАЗОВОГО ПЕРЕХОДА [М]
#

h = 0.03

#
# ПАРАМЕТРЫ ОБЛАСТИ [М]
#

H = 0.1
W = 0.1

#
# ТЕМПЕРАТУРА
#

# НАЧАЛЬНАЯ ТЕМПЕРАТУРА ЛЬДА
T_ice = 253.15  # -20°C
# ТЕМПЕРАТУРА ВОЗДУХА (пока не используется)
T_air = 278.15  # 5°C
# ТЕМПЕРАТУРА ФАЗВОГО ПЕРЕХОДА ВОДА-ЛЁД
T_0 = 273.15  # 0°C


#
# МЕТОД КОНЧЕНЫХ РАЗНОСТЕЙ
#

N_X = 1000  # Число узлов сетки по оси X
N_Y = 1000  # Число узлов сетки по оси Y


t_0 = (c_ice*rho_ice*W**2)/k_ice  # Параметр обезразмеривания времени
end_time = 30/t_0  # Конечное время
dt = 0.1/t_0  # Шаг по времени

N_t = end_time/dt  # Число шагов по времени


gamma = L*rho_w/(t_0*T_0)  # Параметр, используемый в условии Стефана в новых координатах.
