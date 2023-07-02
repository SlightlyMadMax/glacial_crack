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


num = [10.0, 10.021325356076568, 10.030194187423039, 10.036992875609464, 10.042721899286342, 10.047767887390332, 10.052328914183219, 10.056522569886075, 10.060425438344392, 10.064090698542568, 10.067557060786614, 10.070853742912893, 10.07400344094753, 10.077024201670554, 10.07993065586188, 10.08273486034499, 10.085446890588361, 10.088075268637592, 10.090627279077856, 10.093109206875983, 10.095526519490017, 10.0978840084198, 10.100185900724437, 10.102435947942569, 10.104637497776217, 10.106793552448607, 10.108906816650572, 10.110979737253919, 10.113014536460577, 10.115013239664131, 10.116977699019918, 10.118909613507698, 10.120810546099213, 10.122681938531262, 10.124525124075728, 10.126341338630361]


error = []
for i in range(1, 36):
    print(100*abs(num[i]-boundary(ts[i]))/(boundary(ts[i])-10.0))
    error.append(100*abs(num[i]-boundary(ts[i]))/(boundary(ts[i])-10.0))

plt.plot(ts, num, 'r', lw=0.5, label='Численное решение')
plt.plot(ts, boundary(ts), 'k', lw=0.5, label='Аналитическая формула')
plt.legend()
plt.ylim(s0, s0+0.13)
plt.xlim(0, 36)
plt.ylabel('d, m')
plt.xlabel('t, дни')
# plt.savefig('../../graphs/comparison/two_phase/1d.eps', format="eps")

plt.clf()
plt.plot(ts[1:36], error, 'k', lw=2, label='Relative error, %')
plt.legend()

plt.ylim(0, 10)
plt.xlim(0, 36)

plt.ylabel('%')
plt.xlabel('days')
plt.show()
# plt.savefig('../../graphs/error/1d.png')
