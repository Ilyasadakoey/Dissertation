# -*- coding: utf-8 -*-
"""
Kinetik Anpassung der Grimm'schen Alacac3 Daten an Oberflächen und Gasphasen-
Arrhenius-Ausdruck
Am Ende auch Berechnung der Korrelation und Kovarianz

Created on Tue Apr 20 12:00:42 2021

@author: atakan
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, least_squares
import pandas as pd
plt.close('all')
fn ="kineticsAlAcAc_dm_0"
np.random.seed(123)
trials = 57
mu_real = .3 # unknown value in a real experiment
sig_real = .025
dat0 =pd.read_csv("T_XAlacac3_tau_Data.csv", sep=";")
anfang=0
ende=12
all_temp = dat0['Centerline T in K'].values[anfang: ende]
all_times = dat0['Mean Residence Time ms'].values[anfang: ende] / 1000
all_x = (dat0['XAlacac3'] / dat0['XAlacac3'][0]).values[anfang: ende]


def k_arrhenius(A,Ea,T):
    return A*np.exp(-Ea/(8.314*T))

def x_t(t, As, Eas, Ag, Eag):
    wo = np.where(all_times == t)
    temp = all_temp[wo]
    # print(t,wo,temp)
    return np.exp(-t*(k_arrhenius(As, Eas, temp) + k_arrhenius(Ag, Eag, temp)))


def xt_ls(para, t, x, temp):
    As, Eas, Ag, Eag = para
    #Ag=5.8e7
    #Eag = 49800
    x_c = np.exp(-t * (k_arrhenius(As, Eas, temp) + k_arrhenius(Ag, Eag, temp)))
    return x - x_c
    
    


parameter = [33000,14000,5.8e7,49800]

f,ax =plt.subplots(1,1)
ax.plot(all_times,all_x,"bs")
# ax.plot(all_times, x_t(all_times, *parameter))
highT = np.where(all_temp > 450)
loes = curve_fit(x_t, all_times, all_x, p0=parameter)

loesls = least_squares(xt_ls, parameter, loss='linear', method='trf',
                       args=( all_times, all_x, all_temp))
print(loesls.x)
Asl,esl,Agl,egl=loesls.x
ax.plot(all_times, x_t(all_times, *loes[0]), "k:")
ax.plot(all_times, x_t(all_times, *loesls.x), "b:")
ax.set_xlabel(" $T$ / (1/K)" )# sollte hier nicht verweilzeit stehen statt 1/T? IA
ax.set_ylabel("(x/x$_0$)")
As,es,Ag,eg=loes[0]
print("Oberfläche, A: %2.2e, Ea: %2.2e, Gas: A: %2.2e, Ea: %2.2e" %
      (As,es,Ag,eg))
perr = np.sqrt(np.diag(loes[1]))
print("Sigma:",perr)
f.savefig(fn+"x-T.png")

f2,ax2 =plt.subplots(1,1)
ax2.plot(1000 / all_temp, np.log10(-np.log(all_x )/ all_times),"bs")
def l_p(para, col):
    As,es,Ag,eg =para
    ks = np.log10(k_arrhenius(As, es, all_temp))
    kg = np.log10(k_arrhenius(Ag, eg, all_temp))
    k_tot = np.log10(k_arrhenius(Ag, eg, all_temp) + k_arrhenius(As, es, all_temp))
    
    ax2.plot(1000 / all_temp, ks, col+"-")
    ax2.plot(1000/all_temp, kg, col+ ".-") 
    ax2.plot(1000/all_temp,k_tot,col+":") 
    ax2.set_xlabel("1000 / $T$ / (1000/K)" )
    print("Oberfläche, A: %2.2e, Ea: %2.2e, Gas: A: %2.2e, Ea: %2.2e" %
      (As,es,Ag,eg))
l_p(loes[0],"k")
l_p(loesls.x, "b")
ax2.set_ylabel("log$_{10}(k /$s)")
f2.savefig(fn+"Arrhenius.png")


RSS = sum(loesls.fun**2)
NumMeas = len(all_temp)
NumParams = len(parameter)
SER = np.sqrt(RSS/(NumMeas - NumParams))
StdResiduals = np.std(loesls.fun)
# Calculate the Q & R values of the Jacobian matrix
q, r = np.linalg.qr(loesls.jac)
# Calcuate the inverse value of R
#rInv = np.linalg.solve(r, np.identity(r.shape[0]))
rInv= rInv = np.linalg.pinv(r)
# Matrix multiply R Inverse by the transpose of R Inverse
JTJInv = np.matmul(rInv, rInv.transpose())
# Multiply this matrix by the squared regression error (the variance)
CovMatrix = SER**2 * JTJInv

stdev = np.sqrt(np.diag(loes[1]))
print("Stdev:",stdev)
print("\n",CovMatrix)
print(SER)
print("\nCorrelation:")
for i in range (4):
    for j in range(i):
        print(j,i,": %1.3f" % ( CovMatrix[i,j]/stdev[i]/stdev[j]))
    

    
    
    
    