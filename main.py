#
import numpy as np
from scipy.optimize import minimize
from prob import init, simu
from subs import subs
from dsim import dsim
#
def loop():
#
#   Initializations
    [n,m,x_p,x_l,x_u,c_t,f_a,m_k,f_d,sub,mov,asy,exp,aux,glo]=init()
    if glo > 0:
        x_p = np.random.rand(n)*(x_u-x_l) + x_l
    x_k=np.zeros(n,dtype=np.float64); x_1=np.zeros(n,dtype=np.float64)
    x_2=np.zeros(n,dtype=np.float64); x_d=1e0*np.ones(m,dtype=np.float64)
    dg_k=np.zeros((m+1,n),dtype=np.float64); dg_1=np.zeros((m+1,n),dtype=np.float64)
    L_k=np.zeros(n,dtype=np.float64); U_k=np.zeros(n,dtype=np.float64); cnv=0
#
#   Screen output
    if glo==0: print(('\n%3s%14s%9s%11s%11s%11s%11s')%\
        ('k', 'Obj', 'Vio', 'Mov', '|dX|', '||dX||', '|kkt|'))
    for k in range(m_k):
#
#       Simulation: function and gradient values
        [g,dg]=simu(n,m,x_p,aux)
        if f_d == 1: # if finite diff. required
            dg[:]=dsim(n,m,x_p,aux)
        if k == 0: dg_1[:]=dg; x_1[:]=x_p[:]
#
#       Subproblem solve
        x_k[:]=x_p; dg_k[:]=dg
        [x_p,x_d,dx_l,dx_u,L_k,U_k]=subs(sub, n, m, x_k, x_d, x_l, x_u, g, dg, x_1, dg_1, \
        x_2 , L_k, U_k, k, mov, asy, exp, aux)
        x_2[:]=x_1; x_1[:]=x_k; dg_1[:]=dg_k
#
#       Metrics; infinity, Euclidean norm, max KKT viol., and effective move limit
        d_xi=max(abs(x_p-x_k)); d_xe=np.linalg.norm(x_p-x_k); kkt=np.zeros(n); mov_min=1e8
        for i in range(n):
            mov_min=min(mov_min, dx_u[i]-dx_l[i])
            if (x_p[i]-x_l[i])>1e-6 and (x_u[i]-x_p[i])>1e-6: 
                kkt[i]= kkt[i] + dg[0][i]
                for j in range(m): 
                    kkt[i] = kkt[i] + x_d[j]*dg[j+1][i]
        d_kkt=max(abs(kkt))
#
#       Screen output
        if glo == 0: print('%3d%14.3e%9.3f%11.1e%11.1e%11.1e%11.1e'%\
            (k, g[0], max(g[1:]),mov_min,d_xi,d_xe,d_kkt))
#
#       Termination
        if g[1] < 0.001 and g[0] < 1.001*(f_a):
            if glo == 0: print('\nTermination at X = xopt.txt\n'); np.savetxt('xopt.txt',x_p)
            cnv=1; break
        if d_xe < c_t:
            if glo == 0: print('\nTermination at X = xopt.txt\n'); np.savetxt('xopt.txt',x_p)
            cnv=1; break
#
#   If max. iter
    if k == m_k-1:
        if glo == 0: print('\nMax. Iter. at X = xopt.txt\n'); np.savetxt('xopt.txt',x_p)
#
    return g[0], max(g[1:]), cnv, x_p, k
#
def bayes(it,ir):
#
#   See Bolton (2004)
#
    a=1e0
    b=5e0
    a_bar=a+b-1
    b_bar=b-ir-1
    tmp=1e0
    for i in range(it):
        tmp=tmp*(it+i+1+b_bar)/(it+i+1+a_bar)
#
    return 1e0-tmp
#
if __name__ == "__main__":
#
    [n,_,_,_,_,_,_,_,_,_,_,_,_,_,glo]=init()
    if glo == 0:
        [_,_,_,_,_]=loop()
    else:
        print('\nRunning Bayesian global optimization ... ')
        res=[]; fopt=1e8; g=0; kcv=0; kot=0; x_o=np.zeros(n,dtype=np.float64)
        for s in range(glo):
            [f0,vio,cnv,x_p,k]=loop()
            if cnv==1 and vio < 1e-3 and f0 < fopt: fopt=min(fopt, f0); g=s; x_o[:]=x_p
            if cnv==1 and vio < 1e-3: kcv=kcv+k
            res.append([f0,vio,cnv,k]); kot=kot+k
# 
        hit=-1
        for s in range(glo):
            if abs(fopt-res[s][0])/abs(fopt) <= 1e-3 and res[s][1] < 1e-3 and res[s][2] == 1:
                hit=hit+1
#
        if hit <= 0: print('Based on convergence criteria, not a single solution was found.')
        else:
            print('\nTotal number of subproblems (function evaluations)\t:\t%d'%kot)
            print('Total number of subproblems in runs which converged\t:\t%d'%kcv)
            print('Total number of times the best known solution was found\t:\t%d'%hit)
            print('Probability of having found the global optimum\t\t:\t%1.2f\n'%bayes(kcv,hit))
            np.savetxt('xopt.txt',x_o)
            print('Solution written to xopt.txt\n')
#
