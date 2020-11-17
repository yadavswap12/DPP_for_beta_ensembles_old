# -*- coding: utf-8 -*-
"""
Created on Thu Oct 08 12:58:26 2020

@author: yadav
"""

# code to generate kernel K(x,y) = (-1/beta)*(delta sigma(x)/delta V(y)) according to eq.(46) in review 'random matrix theory of quantum transport' by Beenakker

import math
import cmath
import numpy
import contour_integral
import matplotlib    #pylab is submodule in matplotlib

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

data1 = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/V_eff_delta/linear_pot/rho_"+str(rho)+"_delta/gamma_"+str(gamma)+"/density/density_delta_psi_epsi=1e-4_c="+str(c_short_delta)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt",float)
data2 = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/V_eff/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/density_psi_epsi=1e-4_c="+str(c_short)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt",float)
#f_out=file("input_output_files/theta_"+str(theta)+"/V_eff/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/kernel_c="+str(c_short_delta)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt","w")

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
I = numpy.identity(len(K))
A = numpy.linalg.inv(I-K)
L = numpy.dot(K,A)   
#K = numpy.random.rand(3, 3)

#numpy.savetxt("input_output_files/theta_"+str(theta)+"/V_eff/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/kernel.txt", K, delimiter = ' ')  
numpy.savetxt("input_output_files/theta_"+str(theta)+"/V_eff/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/kernel_K_c="+str(c_short_delta)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt", K, delimiter = ' ')
numpy.savetxt("input_output_files/theta_"+str(theta)+"/V_eff/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/kernel_L_c="+str(c_short_delta)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt", L, delimiter = ' ')    
#numpy.save(f_out, K)
