#
import osqp
import numpy as np
from scipy import sparse
#
####################################################################################################
#
# LP solved with OSQP with adaptive move-limit as per SAOR
#
def sub_oslp_aml(n,m,x_k,x_d,x_l,x_u,g,dg,x_1,x_2,k,mov):
#
    mov_rel=mov['mov_rel']
    mov_abs=mov['mov_abs']
#
    dx_l=np.ones(n,dtype=np.float64)
    dx_u=np.ones(n,dtype=np.float64)
#
    ddL=np.ones(n,dtype=np.float64)
#
    for i in range(n):
        factor=0.5
        if k > 1:
            osc = (x_k[i]-x_1[i])*(x_1[i]-x_2[i])/mov_abs/(x_u[i]-x_l[i])
            if osc > 1e-9:
                factor=factor*1.2
            else:
                factor=factor*0.7
        dx_l[i] = max(x_k[i]-factor*mov_abs*(x_u[i]-x_l[i]),x_l[i]) - x_k[i]
        dx_u[i] = min(x_k[i]+factor*mov_abs*(x_u[i]-x_l[i]),x_u[i]) - x_k[i]
#
    J=dg[0]
    ind = np.array(range(n))
    Q=sparse.csc_matrix((ddL, (ind, ind)), shape=(n, n))
    tmp=np.zeros((n,n),dtype=np.float64); np.fill_diagonal(tmp,1e0)
    A=sparse.csc_matrix(np.append(dg[1:],tmp,axis=0))
    u=-g[1:]; l=-np.ones(m,dtype=np.float64)*1e16
    l=np.append(l,dx_l); u=np.append(u,dx_u)
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
