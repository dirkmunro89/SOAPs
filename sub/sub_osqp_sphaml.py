#
import osqp
import numpy as np
from scipy import sparse
#
####################################################################################################
#
# LP solved with OSQP with adaptive move-limit as per SAOR
#
def sub_osqp_sphaml(n,m,x_k,x_d,x_l,x_u,g,dg,x_1,g_1,dg_1,x_2,k,mov):
#
    mov_rel=mov['mov_rel']
    mov_abs=mov['mov_abs']
    mov_fct=mov['mov_fct']
#
    dx_l=np.ones(n,dtype=np.float64)
    dx_u=np.ones(n,dtype=np.float64)
#
    ddL=np.zeros(n,dtype=np.float64)
#
    for i in range(n):
        if k > 1:
            osc = (x_k[i]-x_1[i])*(x_1[i]-x_2[i])/mov_abs/(x_u[i]-x_l[i])
            if osc > 1e-9:
                mov_fct[i]=mov_fct[i]*1.2
            else:
                mov_fct[i]=mov_fct[i]*0.7
        dx_l[i] = max(x_k[i]-mov_fct[i]*mov_abs*(x_u[i]-x_l[i]),x_l[i]) - x_k[i]
        dx_u[i] = min(x_k[i]+mov_fct[i]*mov_abs*(x_u[i]-x_l[i]),x_u[i]) - x_k[i]
#
    mov['mov_fct']=mov_fct
#
    c=np.zeros(m+1,dtype=np.float64)
    dnm=np.linalg.norm(x_1-x_k)**2e0
    if k > 0:
        for j in range(m):
            c[j]=g_1[j]-g[j]
            for i in range(n):
                c[j]=c[j]-dg[j][i]*(x_1[i]-x_k[i])
            c[j]=c[j]/dnm # factor 2 to be checked; incorporate as parameter
#
    for i in range(n):
        ddL[i]=c[0]
        for j in range(m):
            ddL[i]=ddL[i]+c[j+1]*x_d[j]
        ddL[i]=max(ddL[i],0e0)
#
    J=dg[0]; ind = np.array(range(n))
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
