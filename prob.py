#
import numpy as np
from scipy.optimize import minimize
#
####################################################################################################
#
#   Simulation: Objective and constraints function value given x
#               derivatives if possible, else finite differences (f_d)
#
def simu(n,m,x_p,aux):
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
####################################################################################################
#
#   Initialisation: Set problem size and starting point
#
#   Subproblem flags (sub)
#   10  :   MMA
#   11  :   MMA with asymptote adaptation heuristic (as per Svanberg 1987) 
#   12  :   Same as 10, but with constraint relaxation (as per Svanverg 1987)
#   13  :   Same as 11, but with constraint relaxation (as per Svanberg 1987)
#   20  :   CONLIN
#   21  :   CONLIN with adaptive exponent
#   30  :   QCQP reciprocal adaptive
#   31  :   QPLP reciprocal adaptive
#
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
    f_d=0
    c_t=1e-8
    f_a=1.340
    m_k=20
#
    sub=10
#
    mov_abs=-0.1e0
    mov_rel=2e0
#
    asy_fac=3e0/4e0#*1e-6
    asy_adp=1e0/2e0
#
    exp_set=2e0
    exp_min=-6e0
    exp_max=-0.1
#
    mov={'mov_abs': mov_abs, 'mov_rel': mov_rel}
    exp={'exp_set': exp_set, 'exp_min': exp_min, 'exp_max': exp_max}
    asy={'asy_fac': asy_fac,'asy_adp': asy_adp}
    aux={}
#
    return n,m,x_i,x_l,x_u,c_t,f_a,m_k,f_d,sub,mov,asy,exp,aux
#
