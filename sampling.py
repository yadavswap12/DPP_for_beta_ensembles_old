# -*- coding: utf-8 -*-
"""
Created on Fri Oct 09 22:26:35 2020

@author: yadav
"""

# code to implement algorithm 1 for sampling from DPP (see section 2.4 page number 145 in the document 'determinantal point processes for machine learning' by Kulesza and Taskar)

import math
import cmath
import numpy
import contour_integral
import matplotlib    #pylab is submodule in matplotlib
import random

theta = 1.0
rho = 2.0    # V(x)=rho*x. for \theta=1 and V(x)=rho*x, V_eff(x)=2*rho*x/(1+gamma)=rho_eff*x (See our beta ensembles paper)
delta = 0.0001
rho_delta = rho+delta
gamma = float(raw_input("gamma is: "))    # asks for user input from command line. to be used in terminal    
rho_eff_delta = 2.0*rho_delta/(1.0+gamma)
rho_eff = 2.0*rho/(1.0+gamma)
#alpha = 8.0    # V(x)=rho*x+alpha*x**(1.0/alpha), alpha>1
c_delta = theta/rho_eff_delta
c = theta/rho_eff
#c = theta/rho
#c = 0.99598385839782522    # from self-consistent calculation using jouwkowsky_parameters_c_selfconsistent_calculation_CR_problem.py for alpha-log(x) potential.
c_short_delta = c_delta
c_short = c

N = int(raw_input("DPP size is: "))    # asks for user input from command line. to be used in terminal    
N0 = int(raw_input("sample size is: "))    # asks for user input from command line. to be used in terminal    

J = numpy.empty(N0,int)
Y = numpy.empty(N0,int)

L = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/V_eff/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/kernel_L_c="+str(c_short_delta)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt",float)
Lambda, v = numpy.linalg.eig(L) 

# Following is the loop 1 for sampling of a DPP algorithm 1 in Kulesza and Taskar. See notes. 

for i in range(N0):    # function len() on array gives no. of rows of array
    while True:
        m = random.randint(0,N-1)    # generates random integer from 0 to N-1
#        m = random.randrange(N)    # generates random integer from 0 to N-1
        p = random.random()    # generates random floating number from [0.0,1.0)
        if((abs(Lambda[m])/(1.0+abs(Lambda[m])))>=p):
            J[i]=m
            break
# Following is the loop 2 for sampling of a DPP algorithm 1 in Kulesza and Taskar. See notes. 
            
for j in range(N0):
    mod_V = N0-j
    while True:
        m_prime = random.randint(0,N-1)    # generates random integer from 0 to N-1
#        m_prime = random.randrange(N)    # generates random integer from 0 to N-1
        e_m_prime = numpy.zeros(N,float)
        e_m_prime[m_prime] = 1.0
        p_prime = random.random()    # generates random floating number from [0.0,1.0)
        summation = 0
        for k in range(N0):
            volume = (numpy.dot(v[J[k]].T,e_m_prime))**2.0    # .T denotes transpose
            summation = summation + volume
        if((1.0/mod_V)*summation>=p_prime):
            Y[j]=m_prime
            for l in range(N0):
                v[l] = v[l]-(numpy.dot(v[l],e_m_prime))*e_m_prime
            break        
    
numpy.savetxt("input_output_files/theta_"+str(theta)+"/V_eff/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/Y_sampling_DPP_c="+str(c_short_delta)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt", Y, newline='n')
    
    
    
    