# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 08:55:06 2020

@author: yadav
"""

# code to generate kernel K(x,y) = (-1/beta)*(delta sigma(x)/delta V(y)) according to eq.(46) and sampling from DPP (see section 2.4 page number 145) in review 'random matrix theory of quantum transport' by Beenakker

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
beta = 1.0
N = int(raw_input("DPP size is: "))    # asks for user input from command line. to be used in terminal    
#N0 = int(raw_input("sample size is: "))    # asks for user input from command line. to be used in terminal    


data1 = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/V_eff_delta/linear_pot/rho_"+str(rho)+"_delta/gamma_"+str(gamma)+"/density/density_delta_psi_epsi=1e-4_c="+str(c_short_delta)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_9000points.txt",float)
data2 = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/V_eff/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/density_psi_epsi=1e-4_c="+str(c_short)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_9000points.txt",float)
#f_out=file("input_output_files/theta_"+str(theta)+"/V_eff/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/kernel_c="+str(c_short_delta)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt","w")

f_out=file("input_output_files/theta_"+str(theta)+"/V_eff/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/Y_sampling_DPP_c="+str(c_short_delta)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_9000points.txt","w")

delta_sigma = data1[:,1]-data2[:,1]
delta_sigma = delta_sigma.reshape(-1, 1)    #reshape function coverts array of shape (n,) to (n,1). See https://stackoverflow.com/questions/39549331/reshape-numpy-n-vector-to-n-1-vector
delta_sigma=delta_sigma[::-1]    # to reverse the order of the elements of array. See https://www.askpython.com/python/array/reverse-an-array-in-python
delta_V = rho_delta*data1[:,0]-rho*data1[:,0]
one_over_delta_V = 1.0/delta_V
one_over_delta_V = one_over_delta_V.reshape(-1, 1)
one_over_delta_V=one_over_delta_V[::-1]    # to reverse the order of the elements of array. See https://www.askpython.com/python/array/reverse-an-array-in-python

#K = numpy.empty([len(data1[:,0]),len(data1[:,0])],float)
#
#for i in range(0,len(data1[:,0])):    # function len() on array gives no. of rows of array
#    for j in range(0,len(data2[:,0])):    # function len() on array gives no. of rows of array
#        K[i,j] = (-1.0/beta)*(delta_sigma[len(data1[:,0])-1-i]/delta_V[len(data2[:,0])-1-j])
#

K = (-1.0/beta)*numpy.dot(delta_sigma,one_over_delta_V.T)    # .T denotes transpose

#I = numpy.identity(len(K))
#A = numpy.linalg.inv(I-K)
#L = numpy.dot(K,A)   
#K = numpy.random.rand(3, 3)

#numpy.savetxt("input_output_files/theta_"+str(theta)+"/V_eff/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/kernel.txt", K, delimiter = ' ')  
#numpy.savetxt("input_output_files/theta_"+str(theta)+"/V_eff/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/kernel_K_c="+str(c_short_delta)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt", K, delimiter = ' ')
#numpy.savetxt("input_output_files/theta_"+str(theta)+"/V_eff/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/kernel_L_c="+str(c_short_delta)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt", L, delimiter = ' ')    
#numpy.save(f_out, K)

#K = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/V_eff/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/kernel_K_c="+str(c_short_delta)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt",float)
K = (K+K.T)/2.0    # to make kernel K symmteric. Note that according to eq.(43) in beenaker review kernel K is symmetric, but when calculated numerically accrodign to eq.(46) it is slightly assymetric due to numerical error.

print('Kernel K has been computed') 

Lambda_K, v_K = numpy.linalg.eigh(K)    # to find out eigenvalues and eigenvectors of real symmetric matrix. 

Lambda_L = Lambda_K/(Lambda_K-1.0)    # relation between eigenvalues of kernel K and eigenvalues of kernel L.
v_L=v_K    # eigenvectors of kernel K and kernel L are same.

print('eigenvalues and eigenvectors of Kernel L has been computed') 

# Following is the loop 1 for sampling of a DPP algorithm 1 in Kulesza and Taskar. See notes. 

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
