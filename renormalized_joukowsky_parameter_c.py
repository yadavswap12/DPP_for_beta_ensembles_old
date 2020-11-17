# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:45:02 2019

@author: yadav
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 07 16:50:40 2019

@author: yadav
"""

#code for computation of parameter 'c' of Joukowsky transform self consistently for application of NUS paper method to solved example in CR paper. See notebook for details
import math
import cmath
import numpy
import contour_integral

b0=-1.32689E-4    # parameters of polynomial fit for f1(x), f1(x)=b0 + b1x + b2x^2 + b3x^3 + b4x^4 
b1=2.79046   # gamma = 0.75
b2=-0.00215
b3=0.00894
b4=-0.02086
b5=0.02831
b6=-0.02216
b7=0.00927
b8=-0.00161

c0=-0.0015
c1=2.79129

#gamma=0.77 c=0.75964909760813049    #corrected3 iteration11
#gamma=0.78 c=0.75533091405029995    #corrected3 iteration11
#gamma=0.79 c=0.75116396169710087    #corrected3 iteration11
#gamma=0.8 c=0.74692392576962918    #corrected3 iteration11

# 0.45328548471896202

theta = 1.0001
gamma = 0.4
#alpha = 8.0    # V(x)=rho*x+alpha*x**(1.0/alpha), alpha>1
rho= 2.0    # V(x)=rho*x  
c = 0.50005    # from self-consistent calculation using jouwkowsky_parameters_c_selfconsistent_calculation_CR_problem.py for alpha-log(x) potential.
#c = 1.375    # from self-consistent calculation using jouwkowsky_parameters_c_selfconsistent_calculation_CR_problem.py for alpha-log(x) potential.
#c =1.875    # from self-consistent calculation using jouwkowsky_parameters_c_selfconsistent_calculation_CR_problem.py for alpha-log(x) potential.

print c

contr = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/contour/nu_contour_c="+str(c)+"_theta="+str(theta)+"_18000points_iter1.0.txt",float)
# Note that the contour does not depend on parameter 'c' (refer notebook) so we will use the above contour for all different c's.

while True:
        
    def Jc(z):    # joukowsky transformation for hard edge
            return (c*(z+1.0)*(((z+1.0)/z)**(1.0/theta)))
            
    def deriv_Jc(z):    # derivative of joukowsky transformation w.r.t. c
            return (z+1.0)*(((z+1.0)/z)**(1.0/theta))

#    def f1(z):    # See notebook
#            return (1.0/(2.0*(math.pi)*1j))*((b0+b1*Jc(z)+b2*(Jc(z))**2.0+b3*(Jc(z))**3.0+b4*(Jc(z))**4.0)/(z))            

#    def f1(z):    # See notebook
#            return (1.0/(2.0*(math.pi)*1j))*((b0+b1*Jc(z)+b2*(Jc(z))**2+b3*(Jc(z))**3+b4*(Jc(z))**4+b5*(Jc(z))**5+b6*(Jc(z))**6)/(z))

#    def f1(z):    # See notebook
#            return (1.0/(2.0*(math.pi)*1j))*((b0+b1*Jc(z)+b2*(Jc(z))**2+b3*(Jc(z))**3+b4*(Jc(z))**4+b5*(Jc(z))**5+b6*(Jc(z))**6+b7*(Jc(z))**7+b8*(Jc(z))**8)/(z))
            
    def f1(z):    # See notebook
            if (Jc(z)).real<3.5:           
                return (1.0/(2.0*(math.pi)*1j))*((b0+b1*Jc(z)+b2*(Jc(z))**2+b3*(Jc(z))**3+b4*(Jc(z))**4+b5*(Jc(z))**5+b6*(Jc(z))**6+b7*(Jc(z))**7+b8*(Jc(z))**8)/(z))
            else:
                return (1.0/(2.0*(math.pi)*1j))*((c0+c1*Jc(z))/(z))
                
                
    F1 = ((contour_integral.contr_intgl(contr,f1)).real) - (1.0+theta)
    
#    def deriv_f1(z):    # derivative of f1(z) w.r.t. c
#            return (1.0/(2.0*(math.pi)*1j))*((b1*deriv_Jc(z)+2.0*b2*Jc(z)*deriv_Jc(z)+3.0*b3*((Jc(z))**2.0)*deriv_Jc(z)+4.0*b4*((Jc(z))**3.0)*deriv_Jc(z))/(z))  
                           
    
#    def deriv_f1(z):    # derivative of f1(z) w.r.t. c
#            return (1.0/(2.0*(math.pi)*1j))*((b1*deriv_Jc(z)+2.0*b2*Jc(z)*deriv_Jc(z)+3.0*b3*((Jc(z))**2)*deriv_Jc(z)+4.0*b4*((Jc(z))**3)*deriv_Jc(z)+5.0*b5*((Jc(z))**4)*deriv_Jc(z)+6.0*b6*((Jc(z))**5)*deriv_Jc(z))/(z))    

#    def deriv_f1(z):    # derivative of f1(z) w.r.t. c
#            return (1.0/(2.0*(math.pi)*1j))*((b1*deriv_Jc(z)+2.0*b2*Jc(z)*deriv_Jc(z)+3.0*b3*((Jc(z))**2)*deriv_Jc(z)+4.0*b4*((Jc(z))**3)*deriv_Jc(z)+5.0*b5*((Jc(z))**4)*deriv_Jc(z)+6.0*b6*((Jc(z))**5)*deriv_Jc(z)+7.0*b7*((Jc(z))**6)*deriv_Jc(z)+8.0*b8*((Jc(z))**7)*deriv_Jc(z))/(z))    

    def deriv_f1(z):    # derivative of f1(z) w.r.t. c
            if (Jc(z)).real<3.5:
                return (1.0/(2.0*(math.pi)*1j))*((b1*deriv_Jc(z)+2.0*b2*Jc(z)*deriv_Jc(z)+3.0*b3*((Jc(z))**2)*deriv_Jc(z)+4.0*b4*((Jc(z))**3)*deriv_Jc(z)+5.0*b5*((Jc(z))**4)*deriv_Jc(z)+6.0*b6*((Jc(z))**5)*deriv_Jc(z)+7.0*b7*((Jc(z))**6)*deriv_Jc(z)+8.0*b8*((Jc(z))**7)*deriv_Jc(z))/(z))    
            else:
                return (1.0/(2.0*(math.pi)*1j))*((c1*deriv_Jc(z))/(z))    

    deriv_F1 = ((contour_integral.contr_intgl(contr,deriv_f1)).real)                         
    
    c_next = c - (F1/deriv_F1)    
         
    error = c_next - c
    print('error is', error)
    
    c = c_next
    
#    print('c is', c)


    
#    if(numpy.amax(error)<=1e-2):
    if(abs(error)<=(1e-10)):
        break

print('renormalized parameter c is', c)            