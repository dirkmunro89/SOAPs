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
#   index transform to enable copy-paste
#   of functions and gradients from
#   (Fortran) SAOi
#
    x=np.zeros(n+1)
    for i in range(n):
        x[i+1]=x_p[i]
#
    g=np.zeros((1+m),dtype=np.float64)
#
# some often occurring terms
    temp0 = 0.124e0
    temp1 = np.sqrt(1.e0 + x[2]**2)
    temp2 = 8.e0/x[1] + 1.e0/(x[1]*x[2])
    temp3 = 8.e0/x[1] - 1.e0/(x[1]*x[2])
#
# the objective function
    g[0]    = x[1]*temp1
#
# the first constraint
    g[1] = temp0*temp1*temp2 - 1.e0
#
# the second constraint
    g[2] = temp0*temp1*temp3 - 1.e0
#
    dg=np.zeros((1+m,n),dtype=np.float64)
#
# some often occurring terms
    tmp0 = 0.124e0
    tmp1 = np.sqrt(1.e0 + x[2]**2)
    tmp2 = 8.e0/x[1] + 1.e0/(x[1]*x[2])
    tmp3 = 8.e0/x[1] - 1.e0/(x[1]*x[2])
    tmp4 = 2.e0*x[2]
#
# derivatives of the objective function
    dg[0][0] = tmp1
    dg[0][1] = x[1]/(2.e0*tmp1)*tmp4
#
# derivatives of the inequality constraints
    dg[1][0] = -tmp0*tmp1*(8.e0/x[1]**2 + 1.e0/(x[1]**2*x[2]))
    dg[1][1] = tmp0/(2.e0*tmp1)*tmp4*tmp2 - tmp0*tmp1/(x[1]*x[2]**2)
    dg[2][0] = -tmp0*tmp1*(8.e0/x[1]**2 - 1.e0/(x[1]**2*x[2]))
    dg[2][1] = tmp0/(2.e0*tmp1)*tmp4*tmp3 + tmp0*tmp1/(x[1]*x[2]**2)
#
    return [g,dg]
#
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
    n=2
    m=2
    x_i=np.ones(n,dtype=np.float64)
    x_l=np.ones(n,dtype=np.float64)
    x_u=np.ones(n,dtype=np.float64)
#
    x_i[0]=1.5
    x_i[1]=0.5
#
    x_l[0]=0.2
    x_l[1]=0.1
#
    x_u[0]=4.0
    x_u[1]=1.6
#
    f_d=0
    c_t=1e-8
    f_a=1.51
    m_k=20
#
    sub=31
#
    mov_abs=-0.1e0
    mov_rel=2e0
#
    exp_set=2e0
    exp_min=-6e0
    exp_max=-0.1#0.9
#
    asy_fac=1e0/5e0*1e-6
    asy_adp=1e0/2e0
#
    aux={'s_l':0.5,'s_u':0.75}
#
    mov={'mov_abs': mov_abs, 'mov_rel': mov_rel}
    exp={'exp_set': exp_set, 'exp_min': exp_min, 'exp_max': exp_max}
    asy={'asy_fac': asy_fac,'asy_adp': asy_adp}
#
    return n,m,x_i,x_l,x_u,c_t,f_a,m_k,f_d,sub,mov,asy,exp,aux
#
