#
import numpy as np
from scipy.optimize import minimize
#
####################################################################################################
#
# CONLIN: dual subproblem
#
def sub_con_exp(n,m,x_k,x_d,x_l,x_u,g,dg,x_1,dg_1,mov,exp):
#
    mov_rel=mov['mov_rel']
    mov_abs=mov['mov_abs']
    exp_set=exp['exp_set']
    exp_min=exp['exp_min']
    exp_max=exp['exp_max']
#
    dx_l=np.ones(n,dtype=np.float64)
    dx_u=np.ones(n,dtype=np.float64)
#
    if exp_set < 0e0:
        a=np.ones(n,dtype=np.float64)*exp_set
    else:
        a=np.ones(n,dtype=np.float64)*exp_max
        for i in range(n):
            for j in range(m+1):
                a_tmp=1e0+np.log(abs(dg_1[j][i]/dg[j][i]))/np.log(x_1[i]/(x_k[i]+1e-6))
                a[i]=max(min(a[i],a_tmp),exp_min)
#
    for i in range(n):
        if mov_abs < 0e0:
            dx_l[i] = max(x_k[i]/mov_rel,x_l[i])
            dx_u[i] = min(mov_rel*x_k[i],x_u[i])
        else:
            dx_l[i] = max(x_k[i]-mov_abs*(x_u[i]-x_l[i]),x_l[i])
            dx_u[i] = min(x_k[i]+mov_abs*(x_u[i]-x_l[i]),x_u[i])
#
    bds=[[0e0,1e8] for i in range(m)]; tup_bds=tuple(bds)
    sol=minimize(con_exp_dual,x_d,args=(n,m,x_k,g,dg,dx_l,dx_u, a), \
        jac=dcon_exp_dual,method='L-BFGS-B',bounds=tup_bds, options={'disp':False})
#
    x_d[:]=sol.x
    x=x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, a)
#
    return [x,x_d,dx_l,dx_u]
#
# CONLIN EXP.: x in terms of dual variables 
#
def x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, a):
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
            tmpn[j] = tmpn[j] + dg[0][j]#*x_k[j]**2e0
        for i in range(m):
            if dg[i+1][j] > 0e0:
                tmpp[j] = tmpp[j] + dg[i+1][j]*x_d[i]
            else:
                tmpn[j] = tmpn[j] + dg[i+1][j]*x_d[i]#*x_k[j]**2e0
        tmpp[j]=max(tmpp[j],0e0)
        tmpn[j]=min(tmpn[j],-1e-6)
#
    for j in range(n):
        x[j] = min(max(x_k[j]*(-tmpp[j]/tmpn[j])**(1e0/(a[j]-1e0)),dx_l[j]),dx_u[j])
#
    return x
#
# CONLIN EXP.: Dual function value
#
def con_exp_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, a):
#
    x=x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, a)
#
    W = g[0]
    for i in range(m):
        W = W + x_d[i]*g[i+1]
    for j in range(n):
        if dg[0][j] > 0e0:
            W = W + dg[0][j]*(x[j]-x_k[j])
        else:
            W = W + dg[0][j]*(x[j]**a[j]-x_k[j]**a[j])/a[j]*(x_k[j])**(1e0-a[j])
        for i in range(m):
            if dg[i+1][j] > 0e0:
                W = W + dg[i+1][j]*(x[j]-x_k[j])*x_d[i]
            else:
                W = W + dg[i+1][j]*(x[j]**a[j]-x_k[j]**a[j])/a[j]*(x_d[i])*(x_k[j])**(1e0-a[j])
#
    return -W
#
# CONLIN EXP.: Dual gradient
#
def dcon_exp_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, a):
#
    x=x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, a)
#
    dW = np.zeros(m,dtype=np.float64)
#
    for i in range(m):
        dW[i] = dW[i] + g[i+1]
        for j in range(n):
            if dg[i+1][j] > 0e0:
                dW[i] = dW[i] + dg[i+1][j]*(x[j]-x_k[j])
            else:
                dW[i] = dW[i] + dg[i+1][j]*(x[j]**a[j]-x_k[j]**a[j])/a[j]*(x_k[j])**(1e0-a[j])
#
    return -dW
#
