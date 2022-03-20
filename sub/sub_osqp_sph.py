#
import osqp
import numpy as np
from scipy import sparse
#
####################################################################################################
#
# Generalised QP (curv approx): dual subproblem
#
def sub_osqp_sph(n,m,x_k,x_d,x_l,x_u,g,dg,x_1,g_1,dg_1,mov,k):
#
    mov_rel=mov['mov_rel']
    mov_abs=mov['mov_abs']
#
    dx_l=np.ones(n,dtype=np.float64)
    dx_u=np.ones(n,dtype=np.float64)
    a=np.zeros((m+1,n),dtype=np.float64)
#
    ddL=np.zeros(n,dtype=np.float64)
#
    c=np.zeros(m+1,dtype=np.float64)
    dnm=np.linalg.norm(x_1-x_k)**2e0
    if k > 0:
        for j in range(m):
            c[j]=g_1[j]-g[j]
            for i in range(n):
                c[j]=c[j]-dg[j][i]*(x_1[i]-x_k[i])
            c[j]=c[j]/dnm #factor 2 to be checked; incorporate as parameter
#
    print(c)
#
    for i in range(n):
        ddL[i]=c[0]
        for j in range(m):
            ddL[i]=ddL[i]+c[j+1]*x_d[j]
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
    J=dg[0]; ind = np.array(range(n))
    Q=sparse.csc_matrix((ddL, (ind, ind)), shape=(n, n))
    tmp=np.zeros((n,n)); np.fill_diagonal(tmp,1e0)
    A=sparse.csc_matrix(np.append(dg[1:],tmp,axis=0))
    u=-g[1:]; l=-np.ones(m,dtype=np.float64)*1e16
#
    l=np.append(l,dx_l)
    u=np.append(u,dx_u)
#
    prob = osqp.OSQP()
    prob.setup(Q, J, A, l, u,verbose=False)
    res=prob.solve()
#
    if res.info.status != 'solved':
        print('WARNING')
#
    x_d[:]=res.y[:m]
    x=x_k+np.maximum(np.minimum(res.x,dx_u),dx_l)
#
    return [x,x_d,dx_l,dx_u]
#
