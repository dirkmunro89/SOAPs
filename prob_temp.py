#
import numpy as np
from scipy.optimize import minimize
#
####################################################################################################
#
#   Functions: Objective and constraints function value given x
#
def simu(n,m,x_p):
#
#   Blah blah
#
    return [g,dg]
#
#   Initialisation: Set problem size and starting point
#
def init():
#
#   Svanberg 5 variate cantilever
#
    n=1
    m=1
#
    x_i=5e0*np.ones(n,dtype=np.float64)
    x_l=1e-6*np.ones(n,dtype=np.float64)
    x_u=10e0*np.ones(n,dtype=np.float64)
#
    sub=1
    mov=-0.1e0
    mov_rel=2e0
    asy_fac=1e0/16e0
#
    f_a=1.340
#
    return n,m,x_i,x_l,x_u,f_a,sub,mov,mov_rel,asy_fac
#
