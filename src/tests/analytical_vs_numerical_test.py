import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

spd = 60.*60.*24.  # Seconds per day
Lf = 334000.0  # Latent heat of fusion (J/kg)
rho_ice = 918.9
rho_w = 999.84
Ks = spd*2.1  # Conductivity of ice (J/mKs)
cs = 2097.  # Heat capacity of ice (J/kgK) - ** Van der Veen uses 2097 but see Tr and Aschwanden 2012)
ks = Ks*2/((rho_w+rho_ice)*cs)  # Cold ice diffusivity (m2/sec)


# Problem Constants
s0 = 9.0
t0 = 0.
Tm = 0.0
T_ = -10.0


ts = np.arange(0, 36, 1)


def boundary(t):
    return s0 + 2*cs*(Tm - T_)*(ks*t)**.5/(np.pi**.5*Lf)


num = [9.0, 9.020336995303495, 9.028835077169656, 9.035344818055828, 9.040829016594408, 9.045658877562822, 9.050024367427133, 9.054038190204798, 9.0577737156144, 9.061281874837158, 9.064599733159888, 9.067755261593094, 9.070770184421223, 9.073661773717667, 9.07644403101404, 9.079128494135764, 9.081724805147937, 9.084241120698188, 9.086684415244784, 9.089060709577057, 9.091375246028207, 9.093632624862815, 9.095836911859621, 9.097991724165785, 9.100100299515006, 9.102165552530018, 9.10419012088193, 9.106176403394839, 9.108126591698495, 9.11004269666528, 9.11192657060576, 9.113779925991675, 9.115604351321071, 9.11740132462046, 9.119172224984036, 9.120918342480682]


error = []
for i in range(1, 36):
    error.append(100*abs(num[i]-boundary(ts[i]))/(boundary(ts[i])-9.0))

plt.plot(ts, num, 'r', lw=0.5, label='Numerical')
plt.plot(ts, boundary(ts), 'k', lw=0.5, label='Analytical')
plt.legend()
plt.ylim(s0, s0+0.15)
plt.xlim(0, 36)
plt.ylabel('m')
plt.xlabel('days')
plt.savefig('../../graphs/comparison/1d.png')

plt.clf()
plt.plot(ts[1:36], error, 'k', lw=2, label='Relative error, %')
plt.legend()

plt.ylim(0, 10)
plt.xlim(0, 36)

plt.ylabel('%')
plt.xlabel('days')
plt.savefig('../../graphs/error/1d.png')
