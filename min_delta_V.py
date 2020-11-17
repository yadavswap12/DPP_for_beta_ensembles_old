# -*- coding: utf-8 -*-
"""
Created on Fri Oct 09 10:27:54 2020

@author: yadav
"""

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

min_abs_delta_V = numpy.amin(abs(delta_V))
print('minimum delta V is', min_abs_delta_V)

min_abs_delta_sigma = numpy.amin(abs(delta_sigma))
print('minimum delta sigma is', min_abs_delta_sigma)

max_abs_delta_sigma = numpy.amax(abs(delta_sigma))
print('maximum delta sigma is', max_abs_delta_sigma)

matplotlib.pylab.plot(data1[:,0],delta_V)
#matplotlib.pylab.ylim(0,1e-14)
#matplotlib.pylab.xlim(0,0.02)
matplotlib.pylab.show()

matplotlib.pylab.plot(data1[:,0],delta_sigma)
#matplotlib.pylab.ylim(0,1e-7)
#matplotlib.pylab.xlim(0,0.02)
matplotlib.pylab.show()


