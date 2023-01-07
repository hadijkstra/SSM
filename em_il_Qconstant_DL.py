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

sigmaX = 0.03*xkm

T=20*tyear; N=2**16; dt = float(T)/N
t=np.linspace(0,T,N+1)

dW1=np.sqrt(dt)*np.random.randn(1,N)
W1=np.cumsum(dW1)

Xem=np.zeros(N+1); Xem[0] = Xzero

mus0 = 0.0 #0.0000005
mus = np.zeros(N+1)
mus = mus0*np.sin(2*np.pi*t/tyear)

for j in range(1,N+1):
    Winc1=np.sum(dW1[0][range(j-1,j)])
    
    Qt = barQ 
    
    Xem[j]=Xem[j-1]+2*dt*( (Qt/A)*(-1 + C2/((Xem[j-1])**2)) \
                +C3/((Xem[j-1])**3)+C0/Xem[j-1]+ \
                (Qt/A)**2*(C1/Xem[j-1]))+sigmaX*Winc1

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.plot(t/tyear,Xem/xkm,color='black')
plt.xlabel(r'$t$ (years)',fontsize=16); 
plt.ylabel(r'$X$ (km)',fontsize=16,rotation=90)
plt.savefig('../paper/figs/Q-constant_ts_noise_SFB_DL.png',bbox_inches='tight')
plt.show()


plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.hist(Xem/xkm, bins = 100, density=True,color='black')
plt.xlabel(r'$X$ (km)',fontsize=16); 
plt.ylabel(r'frequency',fontsize=16,rotation=90)
plt.savefig('../paper/figs/Q-constant_hist_noise_SFB_DL.png',bbox_inches='tight')
plt.show()


print(np.mean(Xem)/xkm)
print(np.sqrt(np.var(Xem))/xkm)
print(skew(Xem, axis=0, bias=True))
print(kurtosis(Xem, axis=0, bias=True)/xkm)


# # Probability density function  check

nx = 1000
x0 = 10.0*xkm
x1 = 250.0*xkm
x = np.linspace(x0,x1,nx+1)
# print(x)

qda1 = barQ/A
    
ps=np.zeros(nx+1)
def integrand(x, cc0, cc1, cc2, cc3, qda, sigma):
    return np.exp(2*(-cc3/(x**2)-2*qda*x-2*qda*cc2/x+ \
            (qda**2*cc1+2*cc0)*np.log(x))/sigma**2)
    #print(2*(-cc3/(x**2)-2*qda*x)/sigma**2)
    #return np.exp(2*(-cc3/(x**2)-2*qda*x)/sigma**2)
 
pnorm = quad(integrand, x0, x1, args=(C0,C1,C2,C3,qda1,sigmaX))
print(pnorm[0]) 
 
print(C0,C1,C2,C3,qda1,sigmaX)
for i in range(0,nx+1):
    ps[i]=np.exp(2*(-C3/(x[i]**2)-2*qda1*x[i]-2*qda1*C2/x[i]+ \
            (qda1**2*C1+2*C0)*np.log(x[i]))/sigmaX**2)/pnorm[0]
# #    ps[i]=np.exp(2*(-C3/(x[i]**2))/sigmaL**2)

# for i in range(0,nx+1):
#     print(2*(-C3/(x[i]**2))/sigmaL**2)
#     ps[i]=np.exp(2*(-C3/(x[i]**2)-2*qda1*x[i])/sigmaL**2)/pnorm[0]
        
# print(ps)
                 
fig, ax1 = plt.subplots(ncols=1, nrows=1)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim([x0/xkm,x1/xkm])
ax1.plot(x/xkm, ps*xkm, label="PDF",color='red')
ax1.hist(Xem/xkm,bins=200,density=True,color='black')
plt.xlabel(r'$x$ (km)',fontsize=16); 
plt.ylabel(r'$p(x)$',fontsize=16,rotation=90)
plt.savefig('../paper/figs/Q-constant_hist-pdf_noise_SFB_DL.png',bbox_inches='tight')
plt.show() 


  
