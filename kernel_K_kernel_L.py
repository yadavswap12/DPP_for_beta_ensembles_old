# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 10:39:53 2020

@author: yadav
"""

# code to compute kernel K and kernel L from two-point correlation function R.

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

#beta = 1.0

#n=3500
#n = int(raw_input("size of L is: "))    # asks for user input from command line. to be used in terminal    

#L = numpy.random.rand(n, n)
#L = (L + L.T)/2.0

#L = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/V_eff/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/kernel_L_c="+str(c_short_delta)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt",float)
#Lambda, v = numpy.linalg.eig(L)

data1 = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/two_point_correlation_function_R_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt",float)
data2 = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/renormalized_density_psi_method_2_epsi=1e-4_gamma="+str(gamma)+"_theta="+str(theta)+"_rho="+str(rho)+"_18000points_corrected4_iter"+str(iteration)+".txt",float)

#f_out=file("input_output_files/theta_"+str(theta)+"/V_eff/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/Y_sampling_DPP_c="+str(c_short_delta)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_9000points.txt","w")

sigma_full = data2[:,1]
sigma_full = sigma_full.reshape(-1, 1)    #reshape function coverts array of shape (n,) to (n,1). See https://stackoverflow.com/questions/39549331/reshape-numpy-n-vector-to-n-1-vector
sigma = numpy.empty([len(data2[:,0])/100+1],float)
sigma = sigma.reshape(-1, 1)    #reshape function coverts array of shape (n,) to (n,1). See https://stackoverflow.com/questions/39549331/reshape-numpy-n-vector-to-n-1-vector

j=0
for i in range(0,len(data2),100):    # function len() on array gives no. of rows of array
   sigma[j,0]=sigma_full[i,0]
   j += 1

#sigma=sigma[::-1]    # to reverse the order of the elements of array. See https://www.askpython.com/python/array/reverse-an-array-in-python

#delta_sigma = data1[:,1]-data2[:,1]
#delta_sigma = delta_sigma.reshape(-1, 1)    #reshape function coverts array of shape (n,) to (n,1). See https://stackoverflow.com/questions/39549331/reshape-numpy-n-vector-to-n-1-vector
#delta_sigma=delta_sigma[::-1]    # to reverse the order of the elements of array. See https://www.askpython.com/python/array/reverse-an-array-in-python
#delta_V = rho_delta*data1[:,0]-rho*data1[:,0]
#one_over_delta_V = 1.0/delta_V
#one_over_delta_V = one_over_delta_V.reshape(-1, 1)
#one_over_delta_V=one_over_delta_V[::-1]    # to reverse the order of the elements of array. See https://www.askpython.com/python/array/reverse-an-array-in-python

R = data1    # .T denotes transpose
#R = (R+R.T)/2.0    # to make two pint correlation function symmteric. Note that according to eq.(43) in beenaker review two-point correlation function R is symmetric, but when calculated numerically accrodign to eq.(46) it is slightly assymetric due to numerical error.

#matplotlib.pyplot.imshow(R, vmin=numpy.amin(R), vmax=numpy.amax(R)/10000)    # see https://stackoverflow.com/questions/16492830/colorplot-of-2d-array-matplotlib
matplotlib.pyplot.imshow(R, vmin=numpy.amax(R)/100, vmax=numpy.amax(R)/10)    # see https://stackoverflow.com/questions/16492830/colorplot-of-2d-array-matplotlib

matplotlib.pylab.show()    

#A = R-R.T

K = (numpy.dot(sigma,sigma.T)+R)**(0.5)
#K = (K+K.T)/2.0    # to make kernel K symmteric. Note that according to eq.(43) in beenaker review kernel K is symmetric, but when calculated numerically accrodign to eq.(46) it is slightly assymetric due to numerical error.

matplotlib.pyplot.imshow(K, vmin=None, vmax=None)    # see https://stackoverflow.com/questions/16492830/colorplot-of-2d-array-matplotlib
matplotlib.pylab.show()    


print('Kernel K has been computed') 

Lambda_K, v_K = numpy.linalg.eig(K)    # to find out eigenvalues and eigenvectors of real symmetric matrix. 
#Lambda_K, v_K = numpy.linalg.eigh(K)    # to find out eigenvalues and eigenvectors of real symmetric matrix. 

Lambda_L = Lambda_K/(Lambda_K-1.0)    # relation between eigenvalues of kernel K and eigenvalues of kernel L.
v_L=v_K    # eigenvectors of kernel K and kernel L are same.

print('eigenvalues and eigenvectors of Kernel L has been computed') 

#Lambda, v = numpy.linalg.eigh(L)

numpy.savetxt("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/eigenvalues_kernel_L_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt", Lambda_L, newline='\n')
numpy.savetxt("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/kernel_K_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt", K, delimiter = ' ')

#stop = timeit.default_timer()
#print("run time for matrix of size "+str(n)+" is: ", stop - start) 