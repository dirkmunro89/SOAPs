#
import numpy as np
from scipy.optimize import minimize
from prob import simu
#
def dsim(n,m,x_k,aux,out):
#
    g=np.zeros((1+m),dtype=np.float64)
    dg=np.zeros((1+m,n),dtype=np.float64)
#
    x_k1=np.zeros(n,dtype=np.float64)
    x_k0=np.zeros(n,dtype=np.float64)
#
    d_x=1e-3
#
    for i in range(n):
        x_k1[:]=x_k; x_k1[i]=x_k1[i]+d_x
        x_k0[:]=x_k; x_k0[i]=x_k0[i]-d_x
#       [f_k,tmp]=simu(n,m,x_k,aux)
        [f_k1,tmp]=simu(n,m,x_k1,aux,out)
        [f_k0,tmp]=simu(n,m,x_k0,aux,out)
        for j in range(m+1):
            dg[j][i] = (f_k1[j]-f_k0[j])/d_x/2e0
#
    return dg
#
