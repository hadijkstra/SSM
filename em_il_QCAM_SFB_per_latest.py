#EM  Euler-Maruyama method on Stochastic Estuary Model

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import skew, kurtosis
import scipy.integrate as integrate
from scipy.integrate import quad

np.random.seed(100)

# Primary parameters
UT = 0.85 #[m/s] tidal velocity
H = 13 #[m] depht 
B = 2100 # [m] width
A = B*H # [m^2] area
beta = 7.6e-04 #[-] compressibility
g = 9.8 #[ms^2], gravitational acceleration 
socn = 35 #[psu] sea salinity
Sc = 2.2 #dimensionless

# Derived parameters
KH = 0.035*UT*H #[m^2/s] horizontal mixing 
KM = 7.28e-05*UT*H #[m^2/s] vertical viscosity 
KS = KM/Sc #[m^2/s] vertical diffusivity
c = np.sqrt(g*beta*socn*H)  #[m/s] propagation velocity 

print(KH,KM,KS,c)

# Coefficients
C0 = KH #[m^2/s]
C1 = 0.0035*H**2/KS #[s]
C2 = ((0.0275*c*H**2/np.sqrt(KM*KS)))**2 #[m^2] 
C3 = (0.0365**3)*c**4*H**6/(KM**2*KS) #[m^4/s]
 
print(C0,C1,C2,C3) 

tday = 24*3600
tyear = 365*tday 
xkm = 1000
barQ = 893 #[m^3/s]

Xzero = 75 * xkm #[m]
Yzero = 0.1  # initial condition for q 

# Fit of CAM noise parameters
M2 = 2.3600119332051808
M3 = -3.4810531925548354

tau = 20*tday
mu = -1/tau
alp1 = M3**2 + 4*(M2**3)
bet1 = 8*mu*(M2**3) + 2*mu*(M3**2)
gam1 = (M3**2)*(mu**2)
sm2 = (-bet1 + np.sqrt(bet1**2-4*alp1*gam1))/(2*alp1)
sm = np.sqrt(sm2)
sa = np.sqrt(-M2*(2*mu + sm**2)) 
print(sa,sm)

sigmaA = sa
sigmaM = sm
fact = sigmaA/sigmaM
tune = sigmaM
sigmaL = 0.0 #0.005*xkm

#tau = 30*tday
#mu = -1/tau #[1/s] drift in discharge


T=20*tyear; N=2**16; dt = float(T)/N
t=np.linspace(0,T,N+1)


dW1=np.sqrt(dt)*np.random.randn(1,N)
W1=np.cumsum(dW1)

dW2=np.sqrt(dt)*np.random.randn(1,N)
W2=np.cumsum(dW2)

Xem=np.zeros(N+1); Xem[0] = Xzero
Yem=np.zeros(N+1); Yem[0] = Yzero

#mus0 = 0.000001
mus0 = 2.0
mus = np.zeros(N+1)
alp = -4.37e-08
bet =  1.12e-07
mus = mus0*(alp*np.cos(2*np.pi*t/tyear)+bet*np.sin(2*np.pi*t/tyear)) 
#mu = 0.0

for j in range(1,N+1):
    Winc1=np.sum(dW1[0][range(j-1,j)])
    Winc2=np.sum(dW2[0][range(j-1,j)])
    
    Qt = barQ*(1 + Yem[j-1])  
    
    Xem[j]=Xem[j-1]+2*dt*( (Qt/A)*(-1 + C2/((Xem[j-1])**2)) \
                +C3/((Xem[j-1])**3)+C0/Xem[j-1]+ \
                (Qt/A)**2*(C1/Xem[j-1]))+sigmaL*Winc1
    Yem[j]=Yem[j-1]+dt*(mu*Yem[j-1]+ mus[j-1]) + \
           tune*(fact + Yem[j-1])*Winc2
    # log formulation 
    #Yem[j]=Yem[j-1]+dt*(mu - sigmaQ**2/2)*Yem[j-1]+sigmaQ*Winc2


#Check Moments of Y 


M0 = scipy.stats.moment(Yem,moment = 0)
M1 = scipy.stats.moment(Yem,moment = 1)
M2 = scipy.stats.moment(Yem,moment = 2)
M3 = scipy.stats.moment(Yem,moment = 3)
print(M0,M1,M2,M3)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.plot(t/tyear,0.94*Xem/xkm,color='black')
plt.xlabel(r'$t$ (years)',fontsize=16); 
plt.ylabel(r'$X_2$ (km)',fontsize=16,rotation=90)
plt.savefig('../paper/figs/L-full_ts_per_SFB.png',bbox_inches='tight')
#plt.savefig('../paper/figs/L-cam_ts_per_SFB.png',bbox_inches='tight')
plt.show()

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.plot(t/tyear,barQ*(Yem + 1),color='black') 
plt.xlabel(r'$t$ (years)',fontsize=16); 
plt.ylabel(r'$Q$ (m$^3$/s)',fontsize=16,rotation=90)
plt.savefig('../paper/figs/Q-full_ts_per_SFB.png',bbox_inches='tight')
#plt.savefig('../paper/figs/Q-full_ts_per_SFB.png',bbox_inches='tight')
plt.show()

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.hist(0.94*Xem/xkm, bins = 100, density=True,color='black')
plt.xlabel(r'$X_2$ (km)',fontsize=16); 
plt.ylabel(r'frequency',fontsize=16,rotation=90)
plt.savefig('../paper/figs/L-full_hist_per_SFB.png',bbox_inches='tight')
#plt.savefig('../paper/figs/L-cam_hist_per_SFB.png',bbox_inches='tight')
plt.show()

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.hist(barQ*(Yem + 1), bins = 100, density=True,color='black')
plt.xlabel(r'$Q$ (m$^3$/s)',fontsize=16); 
plt.ylabel(r'$frequency$',fontsize=16,rotation=90)
plt.savefig('../paper/figs/Q-full_hist_per_SFB.png',bbox_inches='tight')
#plt.savefig('../paper/figs/Q-cam_hist_per_SFB.png',bbox_inches='tight')
plt.show()

def fitfunction(t, a, b):
    return a*t + b

from scipy.optimize import curve_fit
y_arr = np.log(0.94*Xem/xkm) 
x_arr = np.log(barQ*(Yem + 1))
pars, cov = curve_fit(f=fitfunction, xdata=x_arr, ydata=y_arr, p0=[0, 0], bounds=(-np.inf, np.inf))
#pars, cov = curve_fit(f=fitfunction, xdata=x_arr, ydata=y_arr, p0=[0, 0,0], bounds=(-np.inf, np.inf))
print(pars)

fig = plt.figure
ax = plt.gca()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.scatter(barQ*(Yem + 1),0.94*Xem/xkm,color='black') 
ax.plot(barQ*(Yem + 1),np.exp(fitfunction(np.log(barQ*(Yem + 1)),pars[0],pars[1])))
ax.set_yscale('log')
ax.set_xscale('log')
plt.xticks([100,1000,10000])
plt.yticks([40,60,80,100])
plt.xlabel(r'$Q$ (m$^3$/s)',fontsize=16); 
plt.ylabel(r'$X_2$ (km)',fontsize=16,rotation=90)
plt.savefig('../paper/figs/monismith.png',bbox_inches='tight')
#plt.savefig('../paper/figs/Q-full_ts_per_SFB.png',bbox_inches='tight')
plt.show()




print(np.mean(Xem)/xkm)
print(np.sqrt(np.var(Xem))/xkm)
print(skew(Xem, axis=0, bias=True))
print(kurtosis(Xem, axis=0, bias=True)/xkm)

Qem = barQ*(Yem + 1)
print(np.mean(Qem))
print(np.sqrt(np.var(Qem)))
print(skew(Qem, axis=0, bias=True))
print(kurtosis(Qem, axis=0, bias=True))

  
