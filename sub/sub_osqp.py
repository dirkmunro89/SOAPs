#
import osqp
import numpy as np
from scipy import sparse
#
####################################################################################################
#
# Generalised QP (curv approx): dual subproblem
#
def sub_osqp(n,m,x_k,x_d,x_l,x_u,g,dg,x_1,g_1,dg_1,mov,exp,k):
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
        a=np.ones((m+1,n),dtype=np.float64)*exp_set
    else:
        for i in range(n):
            for j in range(m+1):
                if k <= 1:
                    a[j][i]=-1e0
                else:
                    a_tmp=1e0+np.log((dg_1[j][i]+1e-6)/(dg[j][i]+1e-6))/np.log(x_1[i]/(x_k[i]+1e-6))
                    a[j][i]=max(min(exp_max,a_tmp),exp_min)
#
    c0=np.zeros(n,dtype=np.float64)
    cj=np.zeros((m,n),dtype=np.float64)
    ddL=np.zeros(n,dtype=np.float64)
    for i in range(n):
        c0[i]=(dg[0][i])/x_k[i]*(a[0][i]-1e0)
        for j in range(m):
            cj[j][i]=(dg[j+1][i])/x_k[i]*(a[j+1][i]-1e0)
            ddL[i]=ddL[i]+cj[j][i]*x_d[j]
        ddL[i]=ddL[i]+c0[i]
        ddL[i]=max(ddL[i],0e0)
#
    for i in range(n):
        if mov_abs < 0e0:
            dx_l[i] = max(x_k[i]/mov_rel,x_l[i]) - x_k[i]
            dx_u[i] = min(mov_rel*x_k[i],x_u[i]) - x_k[i]
        else:
            dx_l[i] = max(x_k[i]-mov_abs*(x_u[i]-x_l[i]),x_l[i]) - x_k[i]
            dx_u[i] = min(x_k[i]+mov_abs*(x_u[i]-x_l[i]),x_u[i]) - x_k[i]
#
    ind = np.array(range(n))
    J=dg[0]
    Q=sparse.csc_matrix((ddL, (ind, ind)), shape=(n, n))
    tmp=dg[1:]
    for i in range(n):
        tmp2=np.zeros(n)
        tmp2[i]=1
        tmp=np.append(tmp,[tmp2],axis=0)
    A=sparse.csc_matrix(tmp)
    u=-g[1:]
    l=-np.ones(m,dtype=np.float64)*1e16
#
    l=np.append(l,dx_l)
    u=np.append(u,dx_u)
#
    prob = osqp.OSQP()
    prob.setup(Q, J, A, l, u,verbose=False)
    res=prob.solve()
#
    x_d[:]=res.y[:m]
    x=x_k+res.x
#
    return [x,x_d,dx_l,dx_u]
#
