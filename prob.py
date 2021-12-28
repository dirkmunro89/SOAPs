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
    sub=13
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
