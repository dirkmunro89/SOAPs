#
import numpy as np
from scipy.optimize import minimize
from prob import init, simu
from subs import subs
from dsim import dsim
#
if __name__ == "__main__":
#
#   Initializations
    [n,m,x_p,x_l,x_u,cnv,f_a,kmax,sub,fin_dif,mov,mov_rel,asy_fac,con_exp,s]=init()
    x_k=np.zeros(n,dtype=np.float64) 
    x_1=np.zeros(n,dtype=np.float64)
    x_2=np.zeros(n,dtype=np.float64)
    x_d=1e8*np.ones(m,dtype=np.float64)
    dg_k=np.zeros((m+1,n),dtype=np.float64)
    dg_1=np.zeros((m+1,n),dtype=np.float64)
    L_k=np.zeros(n,dtype=np.float64)
    U_k=np.zeros(n,dtype=np.float64)
#
#   Screen output
    print(''); print(('%3s%14s%9s%11s%11s%11s%11s')%\
        ('k', 'Obj', 'Vio', 'Mov', '|dX|', '||dX||', '|kkt|'))
    for k in range(kmax):
#
#       Simulation: function and gradient values
        [g,dg]=simu(n,m,x_p)
        if fin_dif == 1:
            dg[:]=dsim(n,m,x_p)
        if k == 0: dg_1[:]=dg; x_1[:]=x_p[:]
#
#       Subproblem setup
        x_k[:]=x_p; dg_k[:]=dg
        [x_p,x_d,dx_l,dx_u,L_k,U_k]=subs(sub, n, m, x_k, x_d, x_l, x_u, g, dg, x_1, dg_1, \
            mov, mov_rel, asy_fac, con_exp, k, s, x_2, L_k, U_k)
        x_2[:]=x_1; x_1[:]=x_k; dg_1[:]=dg_k
#
#       Metrics; infinity, Euclidean norm, max KKT viol., and effective move limit
        d_xi=max(abs(x_p-x_k)); d_xe=np.linalg.norm(x_p-x_k); kkt=np.zeros(n)
        for i in range(n):
            if (x_p[i]-x_l[i])>1e-6 and (x_u[i]-x_p[i])>1e-6: 
                kkt[i]= kkt[i] + dg[0][i]
                for j in range(m): 
                    kkt[i] = kkt[i] + x_d[j]*dg[j+1][i]
        d_kkt=max(abs(kkt))
        mov_min=1e8
        if mov < 0e0:
            for i in range(n):
                mov_min=min(mov_min, dx_u[i]-dx_l[i])
        else: mov_min=mov
#
#       Screen output
        print('%3d%14.3e%9.3f%11.1e%11.1e%11.1e%11.1e'%\
            (k, g[0], max(g[1:]),mov_min,d_xi,d_xe,d_kkt))
#
#       Termination
        if g[1] < 0.001 and g[0] < 1.001*(f_a):
            print(''); print('Termination at X =', x_p); print(''); break
        if d_xe < cnv:
            print(''); print('Termination at X =', x_p); print(''); break
#
#   If max. iter
    if k == kmax-1:
        print(''); print('Max. Iter. at X =', x_p); print('')
#
