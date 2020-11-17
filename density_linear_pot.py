# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 10:52:31 2018

@author: yadav
"""

#code for computation of density psi(x) for V(x)=rho*x using eq.(4.24) in CR 

import math
import cmath
import numpy
import contour_integral
import matplotlib    #pylab is submodule in matplotlib

theta = 1.0
rho = 2.0    # V(x)=rho*x. for \theta=1 and V(x)=rho*x, V_eff(x)=2*rho*x/(1+gamma)=rho_eff*x (See our beta ensembles paper)
delta = 0.0001
rho_delta = rho+delta
gamma = 2.8
rho_eff = 2.0*rho_delta/(1.0+gamma)
#rho_eff = 2.0*rho/(1.0+gamma)
#alpha = 8.0    # V(x)=rho*x+alpha*x**(1.0/alpha), alpha>1
c = theta/rho_eff
#c = theta/rho
#c = 0.99598385839782522    # from self-consistent calculation using jouwkowsky_parameters_c_selfconsistent_calculation_CR_problem.py for alpha-log(x) potential.
c_short = c
epsi = 1e-4

data1 = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/V_eff_delta/linear_pot/rho_"+str(rho)+"_delta/gamma_"+str(gamma)+"/contour/nu1_contour_delta_c="+str(c_short)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt",float)
data2 = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/V_eff_delta/linear_pot/rho_"+str(rho)+"_delta/gamma_"+str(gamma)+"/mapping/mapping_output_nu1_delta_c="+str(c_short)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt",float)
data3 = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/V_eff_delta/linear_pot/rho_"+str(rho)+"_delta/gamma_"+str(gamma)+"/contour/nu2_contour_delta_c="+str(c_short)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt",float)
data4 = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/V_eff_delta/linear_pot/rho_"+str(rho)+"_delta/gamma_"+str(gamma)+"/mapping/mapping_output_nu2_delta_c="+str(c_short)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt",float)
contr = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/V_eff_delta/linear_pot/rho_"+str(rho)+"_delta/gamma_"+str(gamma)+"/contour/nu_contour_delta_c="+str(c_short)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt",float)
f_out=file("input_output_files/theta_"+str(theta)+"/V_eff_delta/linear_pot/rho_"+str(rho)+"_delta/gamma_"+str(gamma)+"/density/density_delta_psi_epsi=1e-4_c="+str(c_short)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt","w")

#data1 = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/V_eff/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/contour/nu1_contour_c="+str(c_short)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt",float)
#data2 = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/V_eff/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/mapping/mapping_output_nu1_c="+str(c_short)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt",float)
#data3 = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/V_eff/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/contour/nu2_contour_c="+str(c_short)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt",float)
#data4 = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/V_eff/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/mapping/mapping_output_nu2_c="+str(c_short)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt",float)
#contr = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/V_eff/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/contour/nu_contour_c="+str(c_short)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt",float)
#f_out=file("input_output_files/theta_"+str(theta)+"/V_eff/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/density_psi_epsi=1e-4_c="+str(c_short)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt","w")

psi = (theta/(math.pi*data2[:,0]))*(data1[:,1])

for i in range(0,len(data1)):    # function len() on array gives no. of rows of array
    x1 = data2[i,0]

    f_out.write(str(x1)+" "+str(psi[i])+'\n')
    
f_out.close()    # () at the end is necessary to close the file         

matplotlib.pylab.plot(data2[:,0],psi)
matplotlib.pylab.ylim(0,4)
matplotlib.pylab.show()