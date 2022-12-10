#
# PHYSICAL CONSTANTS
#

# Specific heat capacity of water at 0°C [J/(kg*K)]
c_w = 4217

# Density of water at 0 °C [kg/m^3]
rho_w = 999.84

# Heat conductivity of water at 0°C [W/(m*K)]
k_w = 0.569

# Specific heat capacity of ice at -20°C [J/(kg*K)]
c_ice = 1943

# Density of ice at -20°C [kg/m^3]
rho_ice = 919.4

# Heat conductivity of ice at -20°C [W/(m*K)]
k_ice = 2.39

# Latent heat of ice melting [J/kg]
L = 334000


#
# CRACK
#

h = 0.03
s = 0.07

#
# AREA
#

H = 0.1
W = 0.1

#
# TEMPERATURE
#

# Initial water temperature
T_w = 275.15  # 2°C
# Initial ice temperature
T_ice = 253.15  # -20°C
# Air temperature (for tests)
T_air = 278.15  # 5°C
# Water-ice phase transition temperature
T_0 = 273.15  # 0°C


#
# FINITE DIFFERENCE METHOD
#

N_X = 500
N_Y = 1000


t_0 = (c_ice*rho_ice*W**2)/k_ice

end_time = 10/t_0
dt = 1/t_0
N_t = end_time/dt


gamma = L*rho_w/(t_0*T_0)
