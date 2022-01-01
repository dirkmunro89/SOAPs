#
import numpy as np
from scipy.optimize import minimize
#
####################################################################################################
#
# Generalised exponential QP: dual subproblem
#
def sub_oc_comp2(n,m,x_k,x_d,x_l,x_u,g,dg,mov,exp):
#
    mov_rel=mov['mov_rel']
    mov_abs=mov['mov_abs']
    exp_set=exp['exp_set']
    exp_min=exp['exp_min']
    exp_max=exp['exp_max']
#
    dx_l=np.ones(n,dtype=np.float64)
    dx_u=np.ones(n,dtype=np.float64)
    a=np.zeros(n,dtype=np.float64)
#
    if exp_set < 0e0:
        a=np.ones((m,n),dtype=np.float64)*exp_set
    else:
        for i in range(n):
            a[i]=-1e0
#
    for i in range(n):
        if mov_abs < 0e0:
            dx_l[i] = max(x_k[i]/mov_rel,x_l[i])
            dx_u[i] = min(mov_rel*x_k[i],x_u[i])
        else:
            dx_l[i] = max(x_k[i]-mov_abs*(x_u[i]-x_l[i]),x_l[i])
            dx_u[i] = min(x_k[i]+mov_abs*(x_u[i]-x_l[i]),x_u[i])
#
    bds=[[1e-6,1e8] for i in range(m)]; tup_bds=tuple(bds)
    sol=minimize(con_qpq_dual,x_d,args=(n,m,x_k,g,dg,dx_l,dx_u,a), \
        jac=dcon_qpq_dual,method='L-BFGS-B',bounds=tup_bds, options={'disp':False})
#
    x_d[:]=sol.x
#
    x=x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, a)
#
#   [x,x_d[0]]=bis(n,x_k,g,dg,dx_u,dx_l)
#
    return [x,x_d,dx_l,dx_u]
#
def bis(n,x_k,g,dg,dx_u,dx_l):
#
    l1=0e0
    l2=1e9
#
    x = np.zeros(n,dtype=np.float64)
#
    while (l2-l1)/(l1+l2)>1e-3:
#
        lmid=0.5*(l2+l1)
#
        for i in range(n):
            beta=-dg[0][i]/dg[1][i]*x_k[i]**2e0/lmid
            x[i] = max(min((beta)**(1e0/2e0),dx_u[i]),dx_l[i])
#
        gt=g[1]+np.sum(dg[1]*(x-x_k))
        if gt>0:
            l1=lmid
        else:
            l2=lmid
#
    return x, lmid
#
# Generalised exponential QP: x in terms of dual variables 
#
def x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, a):
#
    x = np.zeros(n,dtype=np.float64)
#
    beta=np.zeros(n,dtype=np.float64)
    for i in range(n):
        beta[i]=-dg[0][i]/dg[1][i]*x_k[i]**2e0/x_d[0]
        x[i] = max(min((beta[i])**(1e0/2e0),dx_u[i]),dx_l[i])
#
    return x
#
# Generalised exponential QP: Dual function value
#
def con_qpq_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, a):
#
    x=x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, a)
#
    W = g[0] + x_d[0]*g[1]
    for i in range(n):
        W = W + (x[i]-x_k[i])*(x_k[i]/(x[i]+1e-6))*dg[0][i]
        W = W + x_d[0]*dg[1][i]*(x[i]-x_k[i])
#
    return -W
#
# Generalised expoential QP: Dual gradient
#
def dcon_qpq_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, a):
#
    x=x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, a)
#
    dW = np.zeros(m,dtype=np.float64)
#
    dW[0]=g[1]
    for i in range(n):
        dW[0] = dW[0] + dg[1][i]*(x[i]-x_k[i])
#
    return -dW
#
