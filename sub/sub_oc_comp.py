#
import numpy as np
from scipy.optimize import minimize
#
####################################################################################################
#
def sub_oc_comp(aux,n,m,x_p,dg,g,mov):
#
    [nelx,nely,volfrac,rmin,penal,ft,Emin,Emax,ndof,KE,H,Hs,iK,jK,edofMat,fixed,free,f,u,im,fig]=aux
#
    x_d=np.zeros(m,dtype=np.float64)
    dx_l=np.ones(n,dtype=np.float64)
    dx_u=np.ones(n,dtype=np.float64)
#
    l1=0
    l2=1e9
    move=mov['mov_abs']
    # reshape to perform vector operations
    xnew=np.zeros(nelx*nely,dtype=np.float64)
#
    xnew[:]=x_p
#
    dx_l[:]=np.maximum(x_p-move,0e0)
    dx_u[:]=np.minimum(x_p+move,1e0)
#
    while (l2-l1)/(l1+l2)>1e-3:
#
        lmid=0.5*(l2+l1)
#
        xnew[:]= np.minimum(np.maximum(x_p*np.sqrt(-dg[0]/dg[1]/lmid),dx_l),dx_u)
#
        gt=g[1]+np.sum((dg[1]*(xnew-x_p)))
#
        if gt>0 :
            l1=lmid
        else:
            l2=lmid
#
    print((g[1]+volfrac*n)/float(n))
#
    x_d[0]=lmid
#
    return [xnew,x_d,dx_l,dx_u]
#
