#
import numpy as np
from scipy.optimize import minimize
from prob import init, simu
from subs import subs
#
if __name__ == "__main__":
#
#   Initializations
    [n,m,x_p,x_l,x_u,f_a,sub,mov,mov_rel,asy_fac]=init()
    x_o=x_p; x_d=1e8*np.ones(m,dtype=np.float64)
#
#   Screen output
    print(''); print(('%3s%14s%9s%11s%11s%11s%11s')%\
        ('k', 'Obj', 'Vio', 'Mov', '|dX|', '||dX||', '|kkt|'))
    for k in range(20):
#
#       Simulation: function and gradient values
        [g,dg]=simu(n,m,x_p)
#
#       Subproblem setup
        x_o[:]=x_p
        [x_p,x_d,dx_l,dx_u]=subs(sub, n, m, x_p, x_d, x_l, x_u, g, dg, mov, mov_rel, asy_fac)
#
#       Metrics; infinity, Euclidean norm, max KKT viol., and effective move limit
        d_xi=max(abs(x_p-x_o)); d_xe=np.linalg.norm(x_p-x_o); kkt=np.zeros(n)
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
#
#   If max. iter
    print(''); print('Max. Iter. at X =', x_p); print('')
#
