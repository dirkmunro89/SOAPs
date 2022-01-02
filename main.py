#
import os
import numpy as np
from scipy.optimize import minimize
from prob import init, simu
from subs import subs
from dsim import dsim
from joblib import Parallel, delayed
#
def loop(s):
#
#   Initializations
    [n,m,x_p,x_l,x_u,c_e,c_i,c_v,f_t,f_a,m_k,f_d,sub,mov,asy,exp,aux,glo]=init()
    if glo > 0: x_p = (np.random.rand(n)*(x_u-x_l) + x_l)#*0.5 + 0.25
    x_k=np.zeros(n,dtype=np.float64); x_1=np.zeros(n,dtype=np.float64)
    x_2=np.zeros(n,dtype=np.float64); x_d=np.ones(m,dtype=np.float64)#*1e-6
    dg_k=np.zeros((m+1,n),dtype=np.float64); dg_1=np.zeros((m+1,n),dtype=np.float64)
    L_k=-np.zeros(n,dtype=np.float64); U_k=np.zeros(n,dtype=np.float64)
    g_k=np.zeros(m+1,dtype=np.float64); g_1=np.zeros(m+1,dtype=np.float64); cnv=0
#
#   Screen output
    if glo==0: print(('\n%3s%14s%9s%13s%17s%11s%11s')%\
        ('k', 'Obj', 'Vio', 'Mov', '|dX|', '||dX||', '|kkt|'))
    for k in range(m_k):
#
#       Simulation: function and gradient values
        [g,dg]=simu(n,m,x_p,aux,s,0)
        if f_d == 1: # if finite diff. required
            dg[:]=dsim(n,m,x_p,aux,s,0)
        if k == 0: dg_1[:]=dg; x_1[:]=x_p
        d_f0=abs(g[0]-g_1[0])/abs(g[0])
#
#       Subproblem solve
        x_k[:]=x_p; dg_k[:]=dg; g_k[:]=g
        [x_p,x_d,dx_l,dx_u,L_k,U_k]=subs(sub, n, m, x_k, x_d, x_l, x_u, g, dg, x_1, dg_1, \
        x_2 , L_k, U_k, k, mov, asy, exp, aux)
        x_2[:]=x_1; x_1[:]=x_k; dg_1[:]=dg_k; g_1[:]=g_k
#
#       Metrics; infinity, Euclidean norm, max KKT viol., and effective move limit
        d_xi=max(abs(x_p-x_k));d_xe=np.linalg.norm(x_p-x_k);kkt=np.zeros(n);mov_min=1e8;mov_max=-1e8
        for i in range(n):
            mov_min=min(mov_min, dx_u[i]-dx_l[i])
            mov_max=max(mov_max, dx_u[i]-dx_l[i])
            if (x_p[i]-x_l[i])>1e-6 and (x_u[i]-x_p[i])>1e-6: 
                kkt[i]= kkt[i] + dg[0][i]
                for j in range(m): 
                    kkt[i] = kkt[i] + x_d[j]*dg[j+1][i]
        d_kkt=max(abs(kkt))
#
#       Screen output
        if glo==0: print('%3d%14.3e%9.3f%11.1e|%5.1e%11.1e%11.1e%11.1e'%\
            (k, g[0], max(g[1:]),mov_min,mov_max,d_xi,d_xe,d_kkt))
#
#       Termination
        if g[1] < c_v and g[0] < (1.+1e-3)*(f_a):
            if glo == 0: 
                print('\nTermination at X = xopt_*.txt\n'); np.savetxt('xopt_%d.txt'%s,x_p)
                print('...based on a priori specified function value at analytic solution\n') 
                break
        if d_xe < c_e and max(g[1:]) < c_v:
            if glo == 0: 
                print('\nTermination at X = xopt_*.txt\n'); np.savetxt('xopt_%d.txt'%s,x_p)
                print('...based on convergence limit and Euclidean norm of last step\n') 
            cnv=1; break
        if d_xi < c_i and max(g[1:]) < c_v:
            if glo == 0: 
                print('\nTermination at X = xopt_*.txt\n'); np.savetxt('xopt_%d.txt'%s,x_p)
                print('...based on convergence limit and Infinity norm of last step\n') 
            cnv=1; break
        if k>1 and d_f0 < f_t and max(g[1:]) < c_v:
            if glo == 0: 
                print('\nTermination at X = xopt_*.txt\n'); np.savetxt('xopt_%d.txt'%s,x_p)
                print('...based on objective significant digit change\n') 
            cnv=1; break
#
#   If max. iter
    if k == m_k-1:
        if glo == 0: print('\nMax. Iter. at X = xopt_*.txt\n'); np.savetxt('xopt_%d.txt'%s,x_p)
#
    [g,dg]=simu(n,m,x_p,aux,s,1)
#
    if glo != 0:
        print('.. done with ',s,g[0],max(g[1:]),cnv,k)
#
    return g[0], max(g[1:]), cnv, x_p, k
#
def bayes(it,ir,a,b):
#
#   See Bolton (2004)
#
#   a=1e0; b=5e2
    a_bar=a+b-1; b_bar=b-ir-1; tmp=1e0
    for i in range(it):
        tmp=tmp*(it+i+1+b_bar)/(it+i+1+a_bar)
#
    return 1e0-tmp
#
if __name__ == "__main__":
#
    [n,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,glo]=init()
    if glo == 0: [_,_,_,_,_]=loop(0)
    else:
#
        print('\nRunning Bayesian global optimization ... ')
        res = Parallel(n_jobs=5)(delayed(loop)(i) for i in range(glo))
        fopt=1e8; g=0; kcv=0; kot=0; c=0; x_o=np.zeros(n,dtype=np.float64)
        print('###')
        for s in range(glo):
#           [f0,vio,cnv,x_p,k]=loop(s)
            f0=res[s][0]; vio=res[s][1]; cnv=res[s][2]; x_p=res[s][3]; k=res[s][4]
            if cnv==1 and vio < 1e-3: 
                kcv=kcv+k; c=c+1
                if f0 < fopt: 
                    fopt=min(fopt, f0); g=s; x_o[:]=x_p
            res.append([f0,vio,cnv,k]); kot=kot+k
            print(s,f0,vio,cnv,k)
# 
        hit=-1; loc=1
        for s in range(glo):
            if abs(fopt-res[s][0])/abs(fopt)<=1e-3 and res[s][1]<1e-3 and res[s][2]==1: hit=hit+1
            else:
                if res[s][1] < 1e-3 and res[s][2] == 1: loc=loc+1
#
        p=1e6 # 1/(TBD; number of local minima)
#
        if hit < 0: print('Based on convergence criteria, not a single solution was found.')
        elif hit == 0: print('Best solution found only once')
        else:
            print('\nTotal number of runs\t\t\t\t\t:\t%6d'%glo)
            print('Total number of converged (also feasible) solutions\t:\t%6d'%c)
            print('Total number of subproblems (function evaluations)\t:\t%6d'%kot)
            print('Total number of subproblems in runs which converged\t:\t%6d'%kcv)
            print('Total number of times the best solution was found\t:\t%6d'%hit)
            print('Probability of having found the global optimum\t\t:\t~%5.2f\n'%bayes(kcv,hit,1,p))
            np.savetxt('xopt_g.txt',x_o)
            print('Solution written to xopt_g.txt\n')
#
