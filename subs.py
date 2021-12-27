#
import numpy as np  
from sub.sub_mma import sub_mma
from sub.sub_mma_rel import sub_mma_rel
from sub.sub_con import sub_con
from sub.sub_con_exp import sub_con_exp
from sub.sub_qpq_exp import sub_qpq_exp
from sub.sub_qpl_exp import sub_qpl_exp
#
def subs(sub, n, m, x_k, x_d, x_l, x_u, g, dg, x_1, dg_1, mov, mov_rel, asy_fac, con_exp):
#
    if sub == 10:
        [x_p,x_d,dx_l,dx_u]=sub_mma(n,m,x_k,x_d,x_l,x_u,g,dg,mov,mov_rel,asy_fac)
    elif sub == 11:
        [x_p,x_d,dx_l,dx_u]=sub_mma_rel(n,m,x_k,x_d,x_l,x_u,g,dg,mov,mov_rel,asy_fac)
    elif sub == 20:
        [x_p,x_d,dx_l,dx_u]=sub_con(n,m,x_k,x_d,x_l,x_u,g,dg,mov,mov_rel)
    elif sub == 21:
        [x_p,x_d,dx_l,dx_u]=sub_con_exp(n,m,x_k,x_d,x_l,x_u,g,dg,mov,mov_rel,con_exp,x_1, dg_1)
    elif sub == 30:
        [x_p,x_d,dx_l,dx_u]=sub_qpq_exp(n,m,x_k,x_d,x_l,x_u,g,dg,mov,mov_rel,con_exp,x_1,dg_1)
    elif sub == 31:
        [x_p,x_d,dx_l,dx_u]=sub_qpl_exp(n,m,x_k,x_d,x_l,x_u,g,dg,mov,mov_rel,con_exp,x_1,dg_1)
    else:
        print('ERROR; subsolver not found')
        stop
#
    return [x_p,x_d,dx_l,dx_u]
#
