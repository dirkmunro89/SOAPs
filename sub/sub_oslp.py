#
import osqp
import numpy as np
from scipy import sparse
#
####################################################################################################
#
# LP solved with OSQP
#
def sub_oslp(n,m,x_k,x_d,x_l,x_u,g,dg,x_1,g_1,dg_1,mov):
#
    mov_rel=mov['mov_rel']
    mov_abs=mov['mov_abs']
#
    dx_l=np.ones(n,dtype=np.float64)
    dx_u=np.ones(n,dtype=np.float64)
#
    ddL=np.zeros(n,dtype=np.float64)
#
    for i in range(n):
        if mov_abs < 0e0:
            dx_l[i] = max(x_k[i]/mov_rel,x_l[i]) - x_k[i]
            dx_u[i] = min(mov_rel*x_k[i],x_u[i]) - x_k[i]
        else:
            dx_l[i] = max(x_k[i]-mov_abs*(x_u[i]-x_l[i]),x_l[i]) - x_k[i]
            dx_u[i] = min(x_k[i]+mov_abs*(x_u[i]-x_l[i]),x_u[i]) - x_k[i]
#
    J=dg[0]
    ind = np.array(range(n))
    Q=sparse.csc_matrix((ddL, (ind, ind)), shape=(n, n))
    tmp=dg[1:]
    for i in range(n):
        tmp2=np.zeros(n)
        tmp2[i]=1
        tmp=np.append(tmp,[tmp2],axis=0)
    A=sparse.csc_matrix(tmp)
    u=-g[1:]; l=-np.ones(m,dtype=np.float64)*1e16
#
    l=np.append(l,dx_l); u=np.append(u,dx_u)
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
