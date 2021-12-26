#
import numpy as np
from scipy.optimize import minimize
#
####################################################################################################
#
# Generalised exponential QP: dual subproblem
#
def sub_qpq_exp(n, m, x_k, x_d, x_l, x_u, g, dg, mov, mov_rel, con_exp, x_1, dg_1):
#
    dx_l=np.ones(n,dtype=np.float64)
    dx_u=np.ones(n,dtype=np.float64)
    a=np.zeros((m+1,n),dtype=np.float64)
#
    if con_exp < 0e0:
        a=np.ones((m,n),dtype=np.float64)*con_exp
    else:
        for i in range(n):
            for j in range(m+1):
                a[j][i]=1.+np.log(dg_1[j][i]/dg[j][i])/np.log(x_1[i]/(x_k[i]+1e-6))
                a[j][i]=max(min(-1e-6,1+np.log(dg_1[j][i]/dg[j][i])/np.log(x_1[i]/(x_k[i]+1e-6))),-6)
#
    c0=np.zeros(n,dtype=np.float64)
    cj=np.zeros((m,n),dtype=np.float64)
    for i in range(n):
#       if dg[0][i] < 0e0:
#           c0[i]=dg[0][i]/x_k[i]*min((a[0][i]-1e0),1.0)
#       else:
#           c0[i]=dg[0][i]/x_k[i]*max((a[0][i]-1e0),1.0)
        c0[i]=-abs(dg[0][i]/x_k[i])*(a[0][i]-1e0)
        for j in range(m):
#           if dg[j][i] < 0e0:
#               cj[j][i]=dg[j][i]/x_k[i]*min((a[j+1][i]-1e0),1.0)
#           else:
#               cj[j][i]=dg[j][i]/x_k[i]*max((a[j+1][i]-1e0),1.0)
            cj[j][i]=-abs(dg[j][i])/x_k[i]*(a[j+1][i]-1e0)
#
    for i in range(n):
        if mov < 0e0:
            dx_l[i] = max(x_k[i]/mov_rel,x_l[i])
            dx_u[i] = min(mov_rel*x_k[i],x_u[i])
        else:
            dx_l[i] = max(x_k[j]-mov*(x_u[j]-x_l[j]),x_l[j])
            dx_u[i] = min(x_k[j]+mov*(x_u[j]-x_l[j]),x_u[j])
#
    bds=[[0e0,1e8] for i in range(m)]; tup_bds=tuple(bds)
    sol=minimize(con_qpq_dual,x_d,args=(n,m,x_k,g,dg,dx_l,dx_u, c0, cj), \
        jac=dcon_qpq_dual,method='L-BFGS-B',bounds=tup_bds, options={'disp':False})
#
    x_d[:]=sol.x
#
    x=x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, c0, cj)
#
    return [x,x_d,dx_l,dx_u]
#
# Generalised exponential QP: x in terms of dual variables 
#
def x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, c0, cj):
#
    x = np.zeros(n,dtype=np.float64)
#
    ddL=np.zeros(n,dtype=np.float64)
    for i in range(n):
        ddL[i]=ddL[i]+c0[i]
        for j in range(m):
            ddL[i]=ddL[i]+cj[j][i]*x_d[j]
#
    tmp=np.zeros(n,dtype=np.float64)
    for i in range(n):
        tmp[i]=tmp[i]+dg[0][i]
        for j in range(m):
            tmp[i]=tmp[i]+x_d[j]*dg[j+1][i]
    for i in range(n):
        x[i] = max(min(x_k[i] - tmp[i]/ddL[i],dx_u[i]),dx_l[i])
#
    return x
#
# Generalised exponential QP: Dual function value
#
def con_qpq_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, c0, cj):
#
    x=x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, c0, cj)
#
    ddL=np.zeros(n,dtype=np.float64)
    for i in range(n):
        ddL[i]=ddL[i]+c0[i]
        for j in range(m):
            ddL[i]=ddL[i]+cj[j][i]*x_d[j]
#
    W = g[0]
    for i in range(n):
        W = W + dg[0][i]*(x[i]-x_k[i]) + ddL[i]/2e0*(x[i]-x_k[i])**2e0
        for j in range(m):
            W = W + x_d[j]*(g[j+1] + dg[j+1][i]*(x[i]-x_k[i]))
#
    return -W
#
# Generalised expoential QP: Dual gradient
#
def dcon_qpq_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, c0, cj):
#
    x=x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, c0, cj)
#
    dW = np.zeros(m,dtype=np.float64)
#
    for j in range(m):
        dW[j] = dW[j] + g[j+1]
        for i in range(n):
            dW[j] = dW[j] + dg[j+1][i]*(x[i]-x_k[i]) + cj[j][i]/2e0*(x[i]-x_k[i])**2e0
#
    return -dW
#
