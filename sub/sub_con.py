#
import numpy as np
from scipy.optimize import minimize
#
####################################################################################################
#
# CONLIN: dual subproblem
#
def sub_con(n, m, x_k, x_d, x_l, x_u, g, dg, mov, mov_rel):
#
    dx_l=np.ones(n,dtype=np.float64)
    dx_u=np.ones(n,dtype=np.float64)
#
    for i in range(n):
        if mov < 0e0:
            dx_l[i] = max(x_k[i]/mov_rel,x_l[i])
            dx_u[i] = min(mov_rel*x_k[i],x_u[i])
        else:
            dx_l[i] = max(x_k[j]+mov*(x_u[j]-x_l[j]),x_l[j])
            dx_u[i] = min(x_k[j]-mov*(x_u[j]-x_l[j]),x_u[j])
#
    bds=[[0e0,1e8] for i in range(m)]; tup_bds=tuple(bds)
    sol=minimize(con_dual,x_d,args=(n,m,x_k,g,dg,dx_l,dx_u), \
        jac=dcon_dual,method='L-BFGS-B',bounds=tup_bds, options={'disp':False})
#
    x_d[:]=sol.x
    x=x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u)
#
    return [x,x_d,dx_l,dx_u]
#
# CONLIN: x in terms of dual variables 
#
def x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u):
#
    x = np.zeros(n,dtype=np.float64)
#
    tmpn = np.zeros(n,dtype=np.float64)
    tmpp = np.zeros(n,dtype=np.float64)
#
    for j in range(n):
        if dg[0][j] > 0e0:
            tmpp[j] = tmpp[j] + dg[0][j]
        else:
            tmpn[j] = tmpn[j] - dg[0][j]#*x_k[j]**2e0
        for i in range(m):
            if dg[i+1][j] > 0e0:
                tmpp[j] = tmpp[j] + dg[i+1][j]*x_d[i]
            else:
                tmpn[j] = tmpn[j] - dg[i+1][j]*x_d[i]#*x_k[j]**2e0
        tmpp[j]=max(tmpp[j],1e-6)
        tmpn[j]=max(tmpn[j],0e0)
#
    for j in range(n):
        x[j] = min(max(np.sqrt(tmpn[j]/tmpp[j])*x_k[j],dx_l[j]),dx_u[j])
#
    return x
#
# CONLIN: Dual function value
#
def con_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u):
#
    x=x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u)
#
    W = g[0]
    for i in range(m):
        W = W + x_d[i]*g[i+1]
    for j in range(n):
        if dg[0][j] > 0e0:
            W = W + dg[0][j]*(x[j]-x_k[j])
        else:
            W = W - dg[0][j]*(1e0/x[j]-1e0/x_k[j])*(x_k[j])**2e0
        for i in range(m):
            if dg[i+1][j] > 0e0:
                W = W + dg[i+1][j]*(x[j]-x_k[j])*x_d[i]
            else:
                W = W - dg[i+1][j]*(1e0/x[j]-1e0/x_k[j])*(x_d[i])*(x_k[j])**2e0
#
    return -W
#
# CONLIN: Dual gradient
#
def dcon_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u):
#
    x=x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u)
#
    dW = np.zeros(m,dtype=np.float64)
#
    for i in range(m):
        dW[i] = dW[i] + g[i+1]
        for j in range(n):
            if dg[i+1][j] > 0e0:
                dW[i] = dW[i] + dg[i+1][j]*(x[j]-x_k[j])
            else:
                dW[i] = dW[i] - dg[i+1][j]*(1e0/x[j]-1e0/x_k[j])*(x_k[j])**2e0
#
    return -dW
#
