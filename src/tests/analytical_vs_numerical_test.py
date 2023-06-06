import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

spd = 60.*60.*24.  # Seconds per day
Lf = 334000.0  # Latent heat of fusion (J/kg)
rho_ice = 916.2  # Ice density at 0°C (kg/m^3)
rho_w = 999.84  # Water density at 0°C (kg/m^3)
Ks = spd*2.1  # Conductivity of ice (J/mKs)
cs = 2097.  # Heat capacity of ice (J/kgK) - ** Van der Veen uses 2097 but see Tr and Aschwanden 2012)
ks = Ks/(rho_ice*cs)  # Cold ice diffusivity (m2/sec)
# ks = 1.02*spd/10**6

# print(f"ks = {ks*10**6/spd}")

# Problem Constants
s0 = 10.0
t0 = 0.
Tm = 0.0
T_ = -10.0


ts = np.arange(0, 36, 1)


def boundary(t):
    return s0 + 2*cs*(Tm - T_)*(ks*t)**.5/(np.pi**.5*Lf)


num = [10.0, 10.021331211751948, 10.030202956184954, 10.037003847985606, 10.042734716387786, 10.047782322692264, 10.052344808083769, 10.056539802159428, 10.06044391423777, 10.064110340900713, 10.06757780520912, 10.070875534593112, 10.074026232507961, 10.077047951626358, 10.079955327494442, 10.082760420855463, 10.085473310447991, 10.088102521078099, 10.090655339687949, 10.09313805327541, 10.095556131062338, 10.097914366093452, 10.100216986788631, 10.102467745893215, 10.104669992184755, 10.106826728851082, 10.108940661450434, 10.111014237639381, 10.113049680331457, 10.115049015568546, 10.117014096099792, 10.118946621447122, 10.120848155081573, 10.122720139201293, 10.124563907503193, 10.126380696280137]


error = []
for i in range(1, 36):
    print(100*abs(num[i]-boundary(ts[i]))/(boundary(ts[i])-10.0))
    error.append(100*abs(num[i]-boundary(ts[i]))/(boundary(ts[i])-10.0))

plt.plot(ts, num, 'r', lw=0.5, label='Численное решение')
plt.plot(ts, boundary(ts), 'k', lw=0.5, label='Полуэмпирическая формула')
plt.legend()
plt.ylim(s0, s0+0.13)
plt.xlim(0, 36)
plt.ylabel('d, м')
plt.xlabel('t, дни')
plt.savefig('../../graphs/1d1p.png')
# plt.savefig('../../graphs/comparison/two_phase/1d.eps', format="eps")
plt.show()

# plt.clf()
# plt.plot(ts[1:36], error, 'k', lw=2, label='Relative error, %')
# plt.legend()
#
# plt.ylim(0, 10)
# plt.xlim(0, 36)
#
# plt.ylabel('%')
# plt.xlabel('days')
# plt.savefig('../../graphs/error/1d.png')
