#
import numpy as np
from scipy.optimize import minimize
#
####################################################################################################
#
# MMA: dual subproblem customised for TABLE 3 reproduction, Svanberg 1987
#
def sub_usr(n,m,x_k,x_d,x_l,x_u,g,dg,x_1,x_2,L_k,U_k,k,mov,asy,aux):
#
    mov_rel=mov['mov_rel']
    asy_fac=asy['asy_fac']
#
    d_scl=1e2
    L=np.zeros(n,dtype=np.float64)
    U=np.zeros(n,dtype=np.float64)
    dx_l=np.ones(n,dtype=np.float64)
    dx_u=np.ones(n,dtype=np.float64)
#
    L[0]=asy_fac*x_k[0]
    U[0]=x_k[0]/asy_fac
#
    s_l=aux['s_l']
    s_u=aux['s_u']
#
    i=1
    if k <= 1:
        L[i]=x_k[i] - (x_u[i] - x_l[i])
        U[i]=x_k[i] + (x_u[i] - x_l[i])
    else:
        if (x_k[i]-x_1[i])*(x_1[i]-x_2[i]) < 0e0:
            L[i] = x_k[i] - s_l*(x_1[i] - L_k[i])
            U[i] = x_k[i] + s_l*(U_k[i] - x_1[i])
        else:
            L[i] = x_k[i] - (x_1[i] - L_k[i])/s_u
            U[i] = x_k[i] + (U_k[i] - x_1[i])/s_u
#
    L[0]=max(min(0.4*x_k[0],L[0]),-50.*x_k[0])
    U[0]=max(min(50.*x_k[0],U[0]),2.5e0*x_k[0])
    L[1]=max(min(0.4*x_k[1],L[1]),-50.*x_k[1])
    U[1]=max(min(50.*x_k[1],U[1]),2.5e0*x_k[1])
#
#   ACTIVATE TO REPRODUCE SLP RESULTS; can not figure out the 'w/o move limits case'
#
#   L[0]=-1e6; L[1]=-1e6; U[0]=1e6; U[1]=1e6
#
    for i in range(n):
        dx_l[i] = max(max(x_k[i]/mov_rel, 1.01*L[i]),x_l[i])
        dx_u[i] = min(min(mov_rel*x_k[i], 0.99*U[i]),x_u[i])
#
    r = np.zeros((m+1),dtype=np.float64)
    p = np.zeros((m+1,n),dtype=np.float64)
    q = np.zeros((m+1,n),dtype=np.float64)
    for i in range(m+1):
        r[i] = g[i]
        for j in range(n):
            if dg[i][j] > 0e0:
                p[i][j] = dg[i][j]*(U[j]-x_k[j])**2e0; q[i][j] = 0e0
            else:
                q[i][j] = -dg[i][j]*(x_k[j]-L[j])**2e0; p[i][j] = 0e0
            r[i] = r[i] - p[i][j]/(U[j]-x_k[j]) - q[i][j]/(x_k[j]-L[j])
#
    d=d_scl*np.ones(m,dtype=np.float64)
#
    bds=[[0e0,1e8] for i in range(m)]; tup_bds=tuple(bds)
    sol=minimize(mma_dual,x_d,args=(n,m,r,p,q,dx_l,dx_u,L,U,d), \
        jac=dmma_dual,method='L-BFGS-B',bounds=tup_bds, options={'disp':False})
#
    x_d[:]=sol.x
    [x,z]=x_dual(x_d, n, m, r, p, q, dx_l, dx_u, L, U, d)
#
    return [x,x_d,dx_l,dx_u,L,U]
#
# MMA: x in terms of dual variables 
#
def x_dual(x_d, n, m, r, p, q, dx_l, dx_u, L, U, d):
#
    x = np.zeros(n,dtype=np.float64)
    z = np.zeros(m,dtype=np.float64)
    tmp1 = np.zeros(n,dtype=np.float64)
    tmp2 = np.zeros(n,dtype=np.float64)
#
    for j in range(n):
        tmp1[j]=p[0][j]; tmp2[j]=q[0][j]
        for i in range(m):
            tmp1[j]=tmp1[j]+p[i+1][j]*x_d[i]; tmp2[j]=tmp2[j]+q[i+1][j]*x_d[i]
        tmp1[j]=max(tmp1[j],0e0); tmp2[j]=max(tmp2[j],0e0)
#
    for j in range(n):
        x[j] = (np.sqrt(tmp1[j])*L[j]+np.sqrt(tmp2[j])*U[j])/(np.sqrt(tmp1[j])+np.sqrt(tmp2[j]))
        x[j] = min(max(x[j],dx_l[j]),dx_u[j])
#
    for i in range(m):
        z[i]=(x_d[i]-d[i])/2e0/d[i]
        z[i]=max(z[i],0e0)
#       z[i]=0e0
#
    return x, z
#
# MMA: Dual function value
#
def mma_dual(x_d, n, m, r, p, q, dx_l, dx_u, L, U, d):
#
    [x,z]=x_dual(x_d, n, m, r, p, q, dx_l, dx_u, L, U, d)
#
    tmp11=0e0; tmp22=0e0
    for j in range(n):
        tmp11=tmp11+p[0][j]/(U[j]-x[j]); tmp22=tmp22+q[0][j]/(x[j]-L[j])
        for i in range(m):
            tmp11=tmp11+p[i+1][j]*x_d[i]/(U[j]-x[j]); tmp22=tmp22+q[i+1][j]*x_d[i]/(x[j]-L[j])
#
    W = r[0] + tmp11 + tmp22
    for i in range(m):
        W = W + d*z[i] + d*z[i]**2e0
        W = W - x_d[i]*(0e0-r[i+1] + z[i])
#
    return -W
#
# MMA: Dual gradient
#
def dmma_dual(x_d, n, m, r, p, q, dx_l, dx_u, L, U, d):
#
    [x,z]=x_dual(x_d, n, m, r, p, q, dx_l, dx_u, L, U, d)
#
    tmp11=np.zeros(m); tmp22=np.zeros(m)
    for i in range(m):
        for j in range(n):
            tmp11[i]=tmp11[i]+p[i+1][j]/(U[j]-x[j]); tmp22[i]=tmp22[i]+q[i+1][j]/(x[j]-L[j])
#
    dW = np.zeros(m,dtype=np.float64)
    for i in range(m):
        dW[i] = dW[i] -(0e0-r[i+1] + z[i]) + tmp11[i] + tmp22[i]
#
    return -dW
#
