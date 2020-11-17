# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 10:09:32 2020

@author: yadav
"""
# code to generate two-point correlation function R(x,y) = (1/beta)*(delta sigma(x)/delta V(y)) according to eq.(4) and (7) in Beenakker, C. W. J., 1993a, Phys. Rev. Lett. 70, 1155.
# order of elements of R is such that R(0,0)=R(x=hard_edge 0,y=hard_edge 0) and R(180,180) = R(x=soft_edge b,y=soft_edge b)

import math
import cmath
import numpy
import contour_integral
import contour_integral_delta
import matplotlib    #pylab is submodule in matplotlib
import random

theta = 1.0001   
gamma = 0.4        
rho = 2.0    
c = 0.35849952882094743    # renormalized c for gamma=0.9 computed from python file 'renormalized_joukowsky_parameter_c.py'
#c_short =     
iteration = 39.0
delta = 0.0001
beta = 1.0

b0 = 1.19197E-5 
b1 = 2.77583
b2 = 0.23458
b3 = -1.44393
b4 = 4.32175
b5 = -6.97132
b6 = 6.19106
b7 = -2.84788
b8 = 0.52978

epsi = 1e-4

data1 = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/contour/nu1_contour_gamma="+str(gamma)+"_theta="+str(theta)+"_18000points_iter"+str(iteration)+".txt",float)
data2 = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/mapping/mapping_output_nu1_gamma="+str(gamma)+"_theta="+str(theta)+"_18000points_iter"+str(iteration)+".txt",float)
data3 = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/contour/nu2_contour_gamma="+str(gamma)+"_theta="+str(theta)+"_18000points_iter"+str(iteration)+".txt",float)
data4 = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/mapping/mapping_output_nu2_gamma="+str(gamma)+"_theta="+str(theta)+"_18000points_iter"+str(iteration)+".txt",float)
contr = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/contour/nu_contour_gamma="+str(gamma)+"_theta="+str(theta)+"_18000points_iter"+str(iteration)+".txt",float)
density = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/renormalized_density_psi_method_2_epsi=1e-4_gamma="+str(gamma)+"_theta="+str(theta)+"_rho="+str(rho)+"_18000points_corrected4_iter"+str(iteration)+".txt",float)
#f_out=file("input_output_files/rho_"+str(rho)+"/theta_"+str(theta)+"/gamma_"+str(gamma)+"/density/renormalized_density_psi_delta_method_2_epsi=1e-4_gamma="+str(gamma)+"_theta="+str(theta)+"_rho="+str(rho)+"_18000points_corrected4_iter"+str(iteration)+".txt","w")

psi = density[:,1]
psi = psi.reshape(-1, 1)    #reshape function coverts array of shape (n,) to (n,1). See https://stackoverflow.com/questions/39549331/reshape-numpy-n-vector-to-n-1-vector
psi=psi[::-1]    # to reverse the order of the elements of array. See https://www.askpython.com/python/array/reverse-an-array-in-python

R = numpy.empty([len(data1[:,0])/100+1,len(data1[:,0])/100+1],float)
delta_V = numpy.empty([len(data1[:,0])/100+1],float)
delta_psi = numpy.empty([len(data1[:,0])/100+1],float)


for j in range(0,len(data1),100):    # function len() on array gives no. of rows of array

    x0 = data4[j,0]
    x0_plus_delta_x = data4[j+1,0]
    delta_x0 = x0_plus_delta_x - x0
    
    delta_f = delta 
    delta_deriv_V = delta/x0
    delta_V[j/100] = delta_deriv_V*delta_x0
    
    for i in range(0,len(data1),100):    # function len() on array gives no. of rows of array
        x_2 = data3[i,0]
        y_2 = data3[i,1]
        x_1 = data1[len(data3)-i-1,0]
        y_1 = data1[len(data3)-i-1,1]
        r1 = math.sqrt(x_1**2.0+y_1**2.0)
        r2 = math.sqrt(x_2**2.0+y_2**2.0)
        x1 = data4[i,0]
#        s = complex(x,y+epsi)
#        s_conj = numpy.conjugate(s)
        s1 = complex(x_1*(1.0+epsi/r1),y_1*(1.0+epsi/r1))
        s_conj1 = numpy.conjugate(s1)
        s2 = complex(x_2*(1.0+epsi/r2),y_2*(1.0+epsi/r2))
        s_conj2 = numpy.conjugate(s2)

        def Jc(z):    # joukowsky transformation for hard edge
                return (c*(z+1.0)*(((z+1.0)/z)**(1.0/theta)))    
    
#        def f(z):    # using first identity in eqn.(3.23) and eqn.(3.21) in C.W.
#            return (-1.0/(4.0*(math.pi**2.0)*x1))*(b0+b1*Jc(z)+b2*(Jc(z))**2.0+b3*(Jc(z))**3.0+b4*(Jc(z))**4.0)*((1.0/(z-s1))-(1.0/(z-s2)))

#        def f(z):    # using first identity in eqn.(3.23) and eqn.(3.21) in C.W.
#            return (-1.0/(4*(math.pi**2)*x1))*(b0+b1*Jc(z)+b2*(Jc(z))**2+b3*(Jc(z))**3+b4*(Jc(z))**4+b5*(Jc(z))**5+b6*(Jc(z))**6)*((1/(z-s1))-(1/(z-s2)))        

        def f(z):    # using first identity in eqn.(3.23) and eqn.(3.21) in C.W.
            return (-1.0/(4*(math.pi**2)*x1))*(b0+b1*Jc(z)+b2*(Jc(z))**2+b3*(Jc(z))**3+b4*(Jc(z))**4+b5*(Jc(z))**5+b6*(Jc(z))**6+b7*(Jc(z))**7+b8*(Jc(z))**8)*((1/(z-s1))-(1/(z-s2)))        

        def f_delta(z):    # using first identity in eqn.(3.23) and eqn.(3.21) in C.W.
            return (-1.0/(4*(math.pi**2)*x1))*(b0+b1*Jc(z)+b2*(Jc(z))**2+b3*(Jc(z))**3+b4*(Jc(z))**4+b5*(Jc(z))**5+b6*(Jc(z))**6+b7*(Jc(z))**7+b8*(Jc(z))**8 + delta)*((1/(z-s1))-(1/(z-s2)))        

        
#     complex(0.667727717428,0.772534506901) in above function lies on contour nu1
#     complex(0.667727717428,0.772534506901+epsi) approaches contour nu1 from above
        psi_delta = contour_integral_delta.contr_intgl_delta(contr,f,f_delta,j)     
        #contour_integral = contour_integral.contr_intgl(contr,f)
        #psi = contour_integral
        delta_psi[i/100] = psi_delta - psi[i,0]
        R[i/100,j/100] = (1.0/beta)*(delta_psi[i/100]/delta_V[j/100])
#        f_out.write(str(x1)+" "+str(psi_delta.real)+'\n')
        print (j,i)

numpy.savetxt("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/two_point_correlation_function_R_old_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt", R, delimiter = ' ')

#f_out.close()    # () at the end is necessary to close the file     
#

 