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
#   aux :   auxilliary data passed from init (also available in subproblem)
#   glo :   sample number if multi-starts selected (else 0)
#   out :   flag used to activate output; called with flag = 1 after termination, else 0
#   
#   g   :   1D array of size m + 1, containing function value of objective (at index 0),
#           followed by constraint function values
#   dg  :   2D array of size (m+1,n), containing gradients of objective wrt design variables,
#           in row 0, followed by the gradients of constraints wrt design variables
#           (not used if finite differences are requested)
#
def simu(n,m,x_p,aux,glo,out):
#
    N=aux[0]
#
    g=np.zeros((1+m),dtype=np.float64)
    dg=np.zeros((1+m,n),dtype=np.float64)
#
    P=50e3
    E=2e7
    L=500
    S=L/N
    smax=14e3
    ymax=2.5
#
    y=0e0
    ya=0e0
#
    for i in range(N):
#
        b=x_p[i]
        h=x_p[N+i]
#       weight objective
        g[0]=g[0]+x_p[i]*x_p[i+N]*S
#       deriv to b
#       dg[0]=x_p[i+N]*S
#       deriv to h
#       dg[0]=x_p[i]*S
#       force moment
        M=P*(L-(i+1)*S+S)
#       second moment of area
        I=(b*h**3e0)/12e0
        dIdb=(h**3e0)/12e0
        dIdh=(3e0*b*h**2e0)/12e0
#       stress
        sig=M*h/2e0/I
        g[i+1]=sig/smax-1e0
#       deriv to b
#       dg[i+1][i]=(-sig/I*dIdb)/smax
#       deriv to h
#       dg[i+1][N+i]=-12e0*M/b/h**3e0
#       geometric constraints
        g[N+i+1]=h-20e0*b
#       deriv to b
#       dg[N+i+1][i]=-20e0
#       deriv to h
#       dg[N+i+1][N+i]=1e0
#
        y=P*S**2e0/E/I/2e0*(L-float(i+1)*S+2e0*S/3e0)+ya*S+y
        ya=P*S/E/I*(L-float(i+1)*S+S/2e0)+ya
#
    g[-1]=y/ymax-1e0
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
#   aux :   auxiliary dictionary / list which may be used for custom parameters 
#           (passed to simu, and the subsolver)
#   glo :   Number of samples in multi-start run
#
#   Subproblem flags (sub)
#
#   1   :   OC (restricted to min. comp. problem)
#   2   :   generalised OC (restricted to min. comp. problem)
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
#   c_e=*   :   convergence threshold in terms of Euclidean norm of step size
#   c_i=*   :   convergence threshold in terms of Infinity norm of step size
#   c_v=*   :   threshold for constraint violation (else no termination)
#   f_t=*   :   threshold for termination on objective value change (not recommended)
#   f_a=*   :   a priori known (analytic) solution; used as termination criteria in some tests
#               (set to large negative number otherwise, as it is not recommended)
#   m_k     :   maximum number of iterations
#
def init():
#
#   Etmans Van der Plaats cantilever beam
#
    N=5
    n=2*N
    m=2*N+1
#
    x_i=np.ones(n,dtype=np.float64)
    x_l=0.1*np.ones(n,dtype=np.float64)
    x_u=1e2*np.ones(n,dtype=np.float64)
#
    x_i[:N]=10.#5.
    x_i[N:]=50.#40.
#
    f_d=1
    c_e=1e-2
    c_i=1e-2
    c_v=1e-2
    f_t=0e0
    f_a=-1e8
    m_k=2
#
    sub=31
#
    glo=0
    cpu=0
#
    mov_abs=1.0e0
    mov_rel=2e0
#
    asy_fac=3e0/4e0  
    asy_adp=1e0/2e0
#
    exp_set=-1.0e0
    exp_min=-1e0
    exp_max=1e0
#
    aux=[N]
#
    mov={'mov_abs': mov_abs, 'mov_rel': mov_rel}
    exp={'exp_set': exp_set, 'exp_min': exp_min, 'exp_max': exp_max}
    asy={'asy_fac': asy_fac,'asy_adp': asy_adp}
#
    return n,m,x_i,x_l,x_u,c_e,c_i,c_v,f_t,f_a,m_k,f_d,sub,mov,asy,exp,aux,glo,cpu
#
