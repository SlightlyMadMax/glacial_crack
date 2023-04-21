import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

spd = 60.*60.*24.  # Seconds per day
Lf = 334000.0  # Latent heat of fusion (J/kg)
rho_ice = 918.9
rho_w = 999.84
Ks = spd*2.1  # Conductivity of ice (J/mKs)
cs = 2097.  # Heat capacity of ice (J/kgK) - ** Van der Veen uses 2097 but see Tr and Aschwanden 2012)
ks = Ks*2/((rho_ice+rho_w)*cs)  # Cold ice diffusivity (m2/sec)


# Problem Constants
s0 = 9.0
t0 = 0.
Tm = 0.0
T_ = -10.0


ts = np.arange(0, 36, 1)


def boundary(t):
    return s0 + 2*cs*(Tm - T_)*(ks*t)**.5/(np.pi**.5*Lf)


num = [9.0, 9.02126808222325, 9.030157370691347, 9.036965988860917, 9.042701383427824, 9.047752002900335, 9.05231664469195, 9.05651325422998, 9.060418614236971, 9.064086025507995, 9.067554278063337, 9.070852644599293, 9.074003860412546, 9.07702600131408, 9.07993372013354, 9.082739090828095, 9.085452202436825, 9.088081587931514, 9.090634540818353, 9.093117353440158, 9.095535499422159, 9.097893775474487, 9.100196413098153, 9.102447167651235, 9.1046493901431, 9.106806085681823, 9.108919961487825, 9.110993466664263, 9.11302882539032, 9.115028064820168, 9.116993038684635, 9.118925447377759, 9.120826855147627, 9.122698704885993, 9.124542330914359, 9.126358970088454]


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
