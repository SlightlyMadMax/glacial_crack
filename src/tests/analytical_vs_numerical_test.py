import numpy as np
import matplotlib.pyplot as plt

spy = 60.*60.*24.  # Seconds per day
Lf = 334000.0  # Latent heat of fusion (J/kg)
rho = 919.4
Ks = spy*2.39  # Conductivity of ice (J/mKs)
cs = 1943.  # Heat capacity of ice (J/kgK) - ** Van der Veen uses 2097 but see Tr and Aschwanden 2012)
ks = Ks/(rho*cs)  # Cold ice diffusivity (m2/sec)


# Problem Constants
s0 = 9.0
t0 = 0.
Tm = 0.0
T_ = -20.0


ts = np.arange(0, 36, 1)


def boundary(t):
    return s0 + 2*cs*(Tm - T_)*(ks*t)**.5/(np.pi**.5*Lf)


num = [9.0, 9.041103980693812, 9.058540849902789, 9.071851153479015, 9.083049422944923, 9.09290451800045, 9.101808131017588, 9.109992036027244, 9.117606855332776, 9.124757017966772, 9.131518452431436, 9.137948416571877, 9.144091355460795, 9.149982588819045, 9.155650736766715, 9.16111937454666, 9.166408196145122, 9.171533854011548, 9.176510578698315, 9.181350645064112, 9.186064729071601, 9.190662185013522, 9.195151263838195, 9.199539287183898, 9.203832787632798, 9.208037622868753, 9.21215906943902, 9.21620190040449, 9.220170450138282, 9.224068668780589, 9.227900168299033, 9.231668261683982, 9.235375996489402, 9.239026183685455, 9.242621422599683, 9.24616412257593]

error = []
for i in range(1, 36):
    error.append(100*abs(num[i]-boundary(ts[i]))/(boundary(ts[i])-9.0))

print(num[35])
print(ts[35])
print(ts[0])
print(boundary(ts[35]))
print(9.092987728561686 - boundary(ts[5]))

plt.plot(ts[1:36], error, 'k', lw=2, label='Relative error, %')
# plt.plot(ts, num, 'r', lw=2, label='Numerical')
plt.legend()

plt.ylim(0, 10)
plt.xlim(0, 36)

plt.ylabel('%')
plt.xlabel('days')
plt.savefig('comparison')
