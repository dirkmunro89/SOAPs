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
#   Initialisation: Set problem size and starting point
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
    sub=1
    mov=-0.1e0
    mov_rel=2e0
    asy_fac=1e0/2e0#*1e-6
    con_exp=2e0
#
    f_a=1.510
    kmax=20
#
    return n,m,x_i,x_l,x_u,f_a,kmax,sub,mov,mov_rel,asy_fac,con_exp
#
