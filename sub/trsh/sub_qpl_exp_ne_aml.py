#
import numpy as np
from scipy.optimize import minimize
#
####################################################################################################
#
# Generalised exponential QP: dual subproblem
#
def sub_qpl_exp_ne_aml(n,m,x_k,x_d,x_l,x_u,g,dg,x_1,x_2,dg_1,mov,exp):
#
    mov_rel=mov['mov_rel']
    mov_abs=mov['mov_abs']
    exp_set=exp['exp_set']
    exp_min=exp['exp_min']
    exp_max=exp['exp_max']
#
    dx_l=np.ones(n,dtype=np.float64)
    dx_u=np.ones(n,dtype=np.float64)
    a=np.zeros((m+1,n),dtype=np.float64)
#
    if exp_set < 0e0:
        a=np.ones((m,n),dtype=np.float64)*con_exp
    else:
        for i in range(n):
            for j in range(m+1):
                a_tmp=1e0+np.log(((dg_1[j][i])/(dg[j][i])))/np.log(x_1[i]/(x_k[i]+1e-6))
                if abs(x_k[i] - x_l[i]) < 1e-3 or abs(x_k[i] - x_u[i]) < 1e-3:
                    a_tmp=1.0e0-1e-4
                elif (x_k[i]-x_1[i])*(x_1[i]-x_2[i]) < 0e0:
                    a_tmp=a_tmp*2e0
                elif (x_k[i]-x_1[i])*(x_1[i]-x_2[i]) > 0e0:
                    a_tmp=a_tmp/2e0
                a[j][i]=max(min(exp_max,a_tmp),exp_min)
#
    c0=np.zeros(n,dtype=np.float64)
    cj=np.zeros((m,n),dtype=np.float64)
    ddL=np.zeros(n,dtype=np.float64)
    for i in range(n):
        c0[i]=dg[0][i]/(x_k[i])*(a[0][i]-1e0)
        for j in range(m):
            cj[j][i]=(dg[j][i])/(x_k[i])*(a[j+1][i]-1e0)
            ddL[i]=ddL[i]+cj[j][i]*x_d[j]
        ddL[i]=ddL[i]+c0[i]
        ddL[i]=max(ddL[i],1e0)
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
    sol=minimize(con_qpl_dual,x_d,args=(n,m,x_k,g,dg,dx_l,dx_u, ddL), \
        jac=dcon_qpl_dual,method='L-BFGS-B',bounds=tup_bds, options={'disp':False})
#
    x_d[:]=sol.x
#
    x=x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, ddL)
#
    return [x,x_d,dx_l,dx_u]
#
# Generalised exponential QP: x in terms of dual variables 
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
    for i in range(n):
        x[i] = max(min(x_k[i] - tmp[i]/ddL[i],dx_u[i]),dx_l[i])
#
    return x
#
# Generalised exponential QP: Dual function value
#
def con_qpl_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, ddL):
#
    x=x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, ddL)
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
def dcon_qpl_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, ddL):
#
    x=x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, ddL)
#
    dW = np.zeros(m,dtype=np.float64)
#
    for j in range(m):
        dW[j] = dW[j] + g[j+1]
        for i in range(n):
            dW[j] = dW[j] + dg[j+1][i]*(x[i]-x_k[i])# + cj[j][i]/2e0*(x[i]-x_k[i])**2e0
#
    return -dW
#
