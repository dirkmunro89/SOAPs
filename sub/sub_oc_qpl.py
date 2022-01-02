#
import numpy as np
from scipy.optimize import minimize
#
####################################################################################################
#
# Generalised exponential QP: dual subproblem
#
def sub_oc_qpl(n,m,x_k,x_d,x_l,x_u,g,dg,mov,exp):
#
    mov_rel=mov['mov_rel']; mov_abs=mov['mov_abs']
    exp_set=exp['exp_set']; exp_min=exp['exp_min']; exp_max=exp['exp_max']
#
    dx_l=np.ones(n,dtype=np.float64); dx_u=np.ones(n,dtype=np.float64)
    a=np.zeros((m+1,n),dtype=np.float64)
#
    for i in range(n):
        for j in range(m+1):
            if j == 0:
                a[j][i]=exp_set
            else:
                a[j][i]=1e0
#
    c0=np.zeros(n,dtype=np.float64);cj=np.zeros((m,n),dtype=np.float64)
    ddL=np.zeros(n,dtype=np.float64)
    for i in range(n):
        c0[i]=dg[0][i]/(x_k[i]+1e-6)*(a[0][i]-1e0)
#       for j in range(m):
#           cj[j][i]=(dg[j][i])/(x_k[i])*(a[j+1][i]-1e0)
#           ddL[i]=ddL[i]+cj[j][i]*x_d[j]
        ddL[i]=max(ddL[i]+c0[i],1e-6)
#
        if mov_abs < 0e0:
            dx_l[i] = max(x_k[i]/mov_rel,x_l[i])
            dx_u[i] = min(mov_rel*x_k[i],x_u[i])
        else:
            dx_l[i] = max(x_k[i]-mov_abs*(x_u[i]-x_l[i]),x_l[i])
            dx_u[i] = min(x_k[i]+mov_abs*(x_u[i]-x_l[i]),x_u[i])
#
    bds=[[0e0,1e8] for i in range(m)]; tup_bds=tuple(bds)
    sol=minimize(con_qpl_dual,x_d,args=(n,m,x_k,g,dg,dx_l,dx_u, ddL), \
        jac=dcon_qpl_dual,method='L-BFGS-B',bounds=tup_bds, options={'disp':False})
#
    x_d[:]=sol.x
#
    x=x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, ddL)
#
    return [x,x_d,dx_l,dx_u]
#
# Generalised QP: x in terms of dual variables 
#
def x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, ddL):
#
    x = np.zeros(n,dtype=np.float64)
#
    tmp=np.zeros(n,dtype=np.float64)
    for i in range(n):
        tmp[i]=tmp[i]+dg[0][i]
        for j in range(m):
            tmp[i]=tmp[i]+x_d[j]*dg[j+1][i]
        x[i] = max(min(x_k[i] - tmp[i]/ddL[i],dx_u[i]),dx_l[i])
#
    return x
#
# Generalised QP: Dual function value
#
def con_qpl_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, ddL):
#
    x=x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, ddL)
#
    W = g[0]
    for j in range(m):
        W = W + x_d[j]*g[j+1]
    for i in range(n):
        W = W + dg[0][i]*(x[i]-x_k[i]) + ddL[i]/2e0*(x[i]-x_k[i])**2e0
        for j in range(m):
            W = W + x_d[j]*(dg[j+1][i]*(x[i]-x_k[i]))
#
    return -W
#
# Generalised QP: Dual gradient
#
def dcon_qpl_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, ddL):
#
    x=x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, ddL)
#
    dW = np.zeros(m,dtype=np.float64)
#
    for j in range(m):
        dW[j] = dW[j] + g[j+1]
        for i in range(n):
            dW[j] = dW[j] + dg[j+1][i]*(x[i]-x_k[i])
#
    return -dW
#
