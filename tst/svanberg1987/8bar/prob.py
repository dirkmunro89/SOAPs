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
    fx=40e3
    fy=20e3
    fz=200e3
    f=np.array([fx,fy,fz])
#
    c=np.zeros((n+1,3),dtype=np.float)
    c[0]=np.array([-250.,-250.,0.]); c[1]=np.array([-250.,250.,0.])
    c[2]=np.array([250.,250.,0.]); c[3]=np.array([250.,-250.,0.])
    c[4]=np.array([0.,0.,375.]); c[5]=np.array([-375.,0.,0.])
    c[6]=np.array([0.,375.,0.]); c[7]=np.array([375.,0.,0.])
    c[8]=np.array([0.,-375.,0.])
#
    d={}
    d[0]=[0,4]; d[1]=[1,4]; d[2]=[2,4]; d[3]=[3,4] 
    d[4]=[5,4]; d[5]=[6,4]; d[6]=[7,4]; d[7]=[8,4]
#
    L=np.zeros(n,dtype=np.float64)
    K=np.zeros((3,3),dtype=np.float64)
    for i in range(n):
#
        L[i]=np.linalg.norm(c[d[i]][1]-c[d[i][0]])
        cx = (c[d[i][0]][0] - c[d[i][1]][0])/L[i]
        cy = (c[d[i][0]][1] - c[d[i][1]][1])/L[i]
        cz = (c[d[i][0]][2] - c[d[i][1]][2])/L[i]
        K=K+x_p[i]/L[i]*np.array([[cx**2.,cx*cy,cx*cz],[cx*cy,cy**2.,cy*cz],[cx*cz,cy*cz,cz**2.]])
#
    u=np.matmul(np.linalg.inv(K),f)
    sig=np.zeros(n,dtype=np.float64)
#
    for i in range(n):
#
        cx = (c[d[i][1]][0] - c[d[i][0]][0])/L[i]
        cy = (c[d[i][1]][1] - c[d[i][0]][1])/L[i]
        cz = (c[d[i][1]][2] - c[d[i][0]][2])/L[i]
#
        sig[i]= 1e0/L[i]*(   cx*u[0] + cy*u[1] + cz*u[2]   )
#
    g=np.zeros((1+m),dtype=np.float64)
    dg=np.zeros((1+m,n),dtype=np.float64)
#
    for i in range(8):
        g[0]=g[0]+x_p[i]*L[i]/128211.
        g[i+1] = sig[i] -100e0
        g[i+1+8] = -sig[i] -100e0
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
    n=8
    m=16
    x_i=np.ones(n,dtype=np.float64)*400e0
    x_l=np.ones(n,dtype=np.float64)*100e0
    x_u=np.ones(n,dtype=np.float64)*1e8
#
#   x_i[0]=880.; x_i[1]=720.; x_i[2]=260.; x_i[3]=520.
#   x_i[4]=100.; x_i[5]=100.; x_i[6]=100.; x_i[7]=100.
#
    f_d=1
    c_t=1e0
    f_a=-1.510e8
    m_k=20
#
    glo=0
#
    sub=11
#
    mov_abs=-0.1e0
    mov_rel=2e0
#
    exp_set=2e0
    exp_min=-6e0
    exp_max=-0.1#0.9
#
    asy_fac=1e0/2e0
    asy_adp=1e0/2e0
#
    mov={'mov_abs': mov_abs, 'mov_rel': mov_rel}
    exp={'exp_set': exp_set, 'exp_min': exp_min, 'exp_max': exp_max}
    asy={'asy_fac': asy_fac,'asy_adp': asy_adp}
    aux={}
#
    return n,m,x_i,x_l,x_u,c_t,f_a,m_k,f_d,sub,mov,asy,exp,aux
