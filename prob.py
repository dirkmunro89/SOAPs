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
    C1=0.0624e0
    C2=1e0
#
    g=np.zeros((1+m),dtype=np.float64)
    g[0]=C1*np.sum(x_p)
    g[1]=61e0/x_p[0]**3e0+37e0/x_p[1]**3e0+19e0/x_p[2]**3e0+7e0/x_p[3]**3e0+1e0/x_p[4]**3e0-C2
#
    dg=np.zeros((1+m,n),dtype=np.float64)
    for i in range(n):
        dg[0][i]=C1
    dg[1][0]=-3e0*61e0/x_p[0]**4e0; dg[1][1]=-3e0*37e0/x_p[1]**4e0; dg[1][2]=-3e0*19e0/x_p[2]**4e0
    dg[1][3]=-3e0*7e0/x_p[3]**4e0; dg[1][4]=-3e0*1e0/x_p[4]**4e0
#
    return [g,dg]
#
#   Initialisation: Set problem size and starting point
#
def init():
#
#   Svanberg 5 variate cantilever
#
    n=5
    m=1
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
    kmax=20
#
    return n,m,x_i,x_l,x_u,f_a,kmax,sub,mov,mov_rel,asy_fac
#
