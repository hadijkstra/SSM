#EM  Euler-Maruyama method on Stochastic Estuary Model

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import skew, kurtosis
import scipy.integrate as integrate
from scipy.integrate import quad

np.random.seed(100)

# Primary parameters
H = 13 #[m] depht 
B = 2100 # [m] width
A = B*H # [m^2] area
KH = 2600 #[m^2/s] horizontal mixing 

# Coefficients
C0 = KH #[m^2/s]
C1 = 0
C2 = 0 
C3 = 0 
 
print(C0,C1,C2,C3) 

tday = 24*3600
tyear = 365*tday 
xkm = 1000
barQ = 893 #[m^3/s]

Xzero = 75 * xkm #[m]
Yzero = 0.1  # initial condition for q 

sigmaQ = 1.0e-04
sigmaL = 0.0

tau = 10*tday
mu = -1/tau #[1/s] drift in discharge


T=20*tyear; N=2**16; dt = float(T)/N
t=np.linspace(0,T,N+1)

dW1=np.sqrt(dt)*np.random.randn(1,N)
W1=np.cumsum(dW1)

dW2=np.sqrt(dt)*np.random.randn(1,N)
W2=np.cumsum(dW2)

Xem=np.zeros(N+1); Xem[0] = Xzero
Yem=np.zeros(N+1); Yem[0] = Yzero

#mus0 = 0.0000005
mus = np.zeros(N+1)
mus0 = 0.0
mus = mus0*np.sin(2*np.pi*t/tyear)

for j in range(1,N+1):
    Winc1=np.sum(dW1[0][range(j-1,j)])
    Winc2=np.sum(dW2[0][range(j-1,j)])
    
    Qt = barQ*(1 + Yem[j-1])  
    
    Xem[j]=Xem[j-1]+2*dt*( (Qt/A)*(-1 + C2/((Xem[j-1])**2)) \
                +C3/((Xem[j-1])**3)+C0/Xem[j-1]+ \
                (Qt/A)**2*(C1/Xem[j-1]))+sigmaL*Winc1
    # additive  noise 
    Yem[j]=Yem[j-1]+dt*mu*Yem[j-1]+sigmaQ*Winc2
    # multiplicative noise 
    #Yem[j]=Yem[j-1]+dt*(mu+mus[j-1])*Yem[j-1]+sigmaQ*Yem[j-1]*Winc2
    # log formulation 
    #Yem[j]=Yem[j-1]+dt*(mu - sigmaQ**2/2)*Yem[j-1]+sigmaQ*Winc2

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.plot(t/tyear,Xem/xkm,color='black')
plt.xlabel(r'$t$ (years)',fontsize=16); 
plt.ylabel(r'$X$ (km)',fontsize=16,rotation=90)
plt.savefig('../paper/figs/L-red_ts_standard_SFB_DL.png',bbox_inches='tight')
plt.show()

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.plot(t/tyear,barQ*(Yem + 1),color='black') 
plt.xlabel(r'$t$ (years)',fontsize=16); 
plt.ylabel(r'$Q$ (m$^3$/s)',fontsize=16,rotation=90)
plt.savefig('../paper/figs/Q-red_ts_standard_SFB_DL.png',bbox_inches='tight')
plt.show()


plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.hist(Xem/xkm, bins = 100, density=True,color='black')
plt.xlabel(r'$X$ (km)',fontsize=16); 
plt.ylabel(r'frequency',fontsize=16,rotation=90)
plt.savefig('../paper/figs/L-red_hist_standard_SFB_DL.png',bbox_inches='tight')
plt.show()

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.hist(barQ*(Yem + 1), bins = 100, density=True,color='black')
plt.xlabel(r'$Q$ (m$^3$/s)',fontsize=16); 
plt.ylabel(r'frequency',fontsize=16,rotation=90)
plt.savefig('../paper/figs/Q-red_hist_standard_SFB_DL.png',bbox_inches='tight')
plt.show()



print(np.mean(Xem)/xkm)
print(np.sqrt(np.var(Xem))/xkm)
print(skew(Xem,  axis = 0 , bias=True))
print(kurtosis(Xem, axis =0, bias=True)/xkm)

Qem = barQ*(Yem + 1)
print(np.mean(Qem))
print(np.sqrt(np.var(Qem)))
print(skew(Qem, axis = 0,bias=True)) 
print(kurtosis(Qem, axis = 0, bias=True))

  
