#
import numpy as np
from scipy.optimize import minimize
#
####################################################################################################
#
#   Simulation: Objective and constraints function value given x
#               derivatives if possible, else finite differences (f_d)
#
#   n   :   number of design variables
#   m   :   number of constraints
#   x_p :   current evaluation point (design variables)
#   
#   g   :   1D array of size m + 1, containing function value of objective (at index 0),
#           followed by constraint function values
#   dg  :   2D array of size (m+1,n), containing gradients of objective wrt design variables,
#           in row 0, followed by the gradients of constraints wrt design variables
#           (not used if finite differences are requested)
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
#
####################################################################################################
#
#   Initialisation: Set problem size and starting point
#
#   n   :   number of design variables
#   m   :   number of constraints
#   x_i :   starting point
#   x_l :   global lower bounds on design variables
#   x_u :   global upper bounds on design variables
#   aux :   auxiliary dictionary which may be used for custom parameters (passing of)
#
#   Subproblem flags (sub)
#
#   10  :   MMA
#   11  :   MMA with asymptote adaptation heuristic (as per Svanberg 1987) 
#   12  :   Same as 10, but with constraint relaxation (as per Svanverg 1987)
#   13  :   Same as 11, but with constraint relaxation (as per Svanberg 1987)
#   20  :   CONLIN
#   21  :   CONLIN with adaptive exponent
#   30  :   QCQP reciprocal adaptive
#   31  :   QPLP reciprocal adaptive
#
#   Suproblem parameters
#
#   10,12   :   asy_fac     :       Simple rule from Svanberg 1987 ('t')
#           :   mov_rel     :       Svanberg 1987 bounds factor (24)
#
#   11,13   :   asy_adp     :       L and U heuristic Svanberg 1987 ('s')
#           :   mov_rel     :       Svanberg 1987 bounds factor (24)
#
#   20      :   mov_abs     :       Move-limit as factor of design domain
#           :   mov_rel     :       if mov_abs is set to < 0, then Svanberg 
#                                   bounds factor is used
#   21      :   mov_*       :       Same as 20
#           :   exp_set     :       set constant exponent (e.g. -2 instead of -1)
#           :   exp_min     :       if exp_set is set to > 0, then adaptive exponential
#                                   fitting is used, requiring ...
#           :   exp_max     :       maximum exponent bound (-0.1 suggested)
#           :   exp_min     :       minimum exponent bound (-6 suggested)
#
#   30,31   :   mov_*       :       Same as 20
#           :   exp_*       :       Same as 21
#
#   Global flags and parameters
#
#   f_d=1   :   activate gradient calculation via finite differences (dont if 0)
#   c_t=*   :   convergence limit in terms of step size (Euclidean norm)
#   f_a=*   :   a priori known (analytic) solution; used in termination criteria (set to large 
#               negative number if not applicable)
#   m_k     :   maximum outer iterations
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
    glo=99
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
    return n,m,x_i,x_l,x_u,c_t,f_a,m_k,f_d,sub,mov,asy,exp,aux,glo
#
