# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 13:04:45 2020

@author: yadav
"""

# code to generate sampling from DPP (see section 2.4 page number 145) in review 'random matrix theory of quantum transport' by Beenakker

import math
import cmath
import numpy
import contour_integral
import matplotlib    #pylab is submodule in matplotlib
import random
import timeit

start = timeit.default_timer()

theta = 1.0001
rho = 2.0    # V(x)=rho*x. for \theta=1 and V(x)=rho*x, V_eff(x)=2*rho*x/(1+gamma)=rho_eff*x (See our beta ensembles paper)
#delta = 0.0001
#rho_delta = rho+delta
gamma = 0.4
#gamma = float(raw_input("gamma is: "))    # asks for user input from command line. to be used in terminal    
#rho_eff_delta = 2.0*rho_delta/(1.0+gamma)
#rho_eff = 2.0*rho/(1.0+gamma)
#alpha = 8.0    # V(x)=rho*x+alpha*x**(1.0/alpha), alpha>1
#c_delta = theta/rho_eff_delta
#c = theta/rho_eff
#c = theta/rho
#c = 0.99598385839782522    # from self-consistent calculation using jouwkowsky_parameters_c_selfconsistent_calculation_CR_problem.py for alpha-log(x) potential.
#c_short_delta = c_delta
#c_short = c

iteration = 39.0

K = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/kernel_K_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt",float)

f_out=file("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/Y_sampling_DPP_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt","w")

Lambda_K, v_K = numpy.linalg.eigh(K)    # to find out eigenvalues and eigenvectors of real symmetric matrix. 

Lambda_L = Lambda_K/(Lambda_K-1.0)    # relation between eigenvalues of kernel K and eigenvalues of kernel L.
v_L=v_K    # eigenvectors of kernel K and kernel L are same.

print('eigenvalues and eigenvectors of Kernel L has been computed') 

N = len(K)   

J = numpy.zeros(N,int)
#Y = numpy.empty(N0,int)

#for i in range(N):    # function len() on array gives no. of rows of array
#    while True:
#        m = random.randint(0,N-1)    # generates random integer from 0 to N-1
##        m = random.randrange(N)    # generates random integer from 0 to N-1
#        p = random.random()    # generates random floating number from [0.0,1.0)
#        if((abs(Lambda_L[m])/(1.0+abs(Lambda_L[m])))>=p):
#            J[i]=m
#            break

for i in range(N):    # function len() on array gives no. of rows of array
    p = random.random()    # generates random floating number from [0.0,1.0)
    if((abs(Lambda_L[i])/(1.0+abs(Lambda_L[i])))>=p):
        J[i]=i



J_masked = numpy.ma.masked_equal(J,0)    # to mask all the zeros of array J
J = J_masked[~J_masked.mask]    # to remove all the masked values of array J_masked

N0 = len(J)
Y = numpy.empty(N0,int)

print('size of the sample is',N0) 

# Following is the loop 2 for sampling of a DPP algorithm 1 in Kulesza and Taskar. See notes. 
            
for j in range(N0):
    mod_V = N0-j
    print('iterations remaining are',N0-j)
    while True:
        m_prime = random.randint(0,N-1)    # generates random integer from 0 to N-1
#        m_prime = random.randrange(N)    # generates random integer from 0 to N-1
        e_m_prime = numpy.zeros(N,float)
        e_m_prime[m_prime] = 1.0
        p_prime = random.random()    # generates random floating number from [0.0,1.0)
        summation = 0
        for k in range(N0):
            volume = (numpy.dot(v_L[J[k]].T,e_m_prime))**2.0    # .T denotes transpose
            summation = summation + volume
        if((1.0/mod_V)*summation>=p_prime):
            Y[j]=m_prime
            f_out.write(str(Y[j])+'\n')

            for l in range(N0):
                v_L[l] = v_L[l]-(numpy.dot(v_L[l],e_m_prime))*e_m_prime
            break        

f_out.close()    # () at the end is necessary to close the file             
#numpy.savetxt("input_output_files/theta_"+str(theta)+"/V_eff/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/Y_sampling_DPP_c="+str(c_short_delta)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_9000points.txt", Y, newline='n')

