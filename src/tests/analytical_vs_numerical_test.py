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


num = [9.0, 9.021272750697298, 9.030120392627207, 9.036902707789592, 9.04261785679124, 9.047651568876432, 9.052201456281447, 9.05638483449633, 9.060278108230627, 9.063934331964113, 9.067392125304774, 9.070680637589545, 9.07382251160735, 9.076835751786316, 9.079734954557571, 9.082532148429753, 9.085237385200047, 9.08785916688745, 9.09040476095468, 9.092880437591456, 9.095291651397932, 9.097643182604052, 9.099939248323382, 9.10218359126719, 9.104379551259218, 9.106530123455633, 9.108638006179278, 9.110705640541221, 9.112735243510487, 9.114728835710242, 9.116688264933423, 9.118615226157296, 9.12051127867218, 9.122377860817062, 9.124216302721457, 9.126027837371971]


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

plt.ylim(0, 1)
plt.xlim(0, 36)

plt.ylabel('%')
plt.xlabel('days')
plt.savefig('../../graphs/error/1d.png')
