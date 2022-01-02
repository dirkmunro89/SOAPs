#
import numpy as np  
from sub.sub_mma import sub_mma
from sub.sub_mma_rel import sub_mma_rel
from sub.sub_con import sub_con
from sub.sub_con_exp import sub_con_exp
from sub.sub_qpq_exp import sub_qpq_exp
from sub.sub_qpl_exp import sub_qpl_exp
from sub.sub_qpl_exp_ne import sub_qpl_exp_ne
from sub.sub_qpl_exp_ne_aml import sub_qpl_exp_ne_aml
from sub.sub_mmaa import sub_mmaa
from sub.sub_mma_rela import sub_mma_rela
from sub.sub_usr import sub_usr
from sub.sub_oc_comp import sub_oc_comp
from sub.sub_oc_exp import sub_oc_exp
from sub.sub_oc_qpl import sub_oc_qpl
#
def subs(sub, n, m, x_k, x_d, x_l, x_u, g, dg, x_1, dg_1, x_2, L_k, U_k, k, mov, asy, exp, aux):
#
    if sub == 1:
        [x_p,x_d,dx_l,dx_u]=sub_oc_comp(aux,n,m,x_k,dg,g,mov)
    elif sub == 2:
        [x_p,x_d,dx_l,dx_u]=sub_oc_exp(n,m,x_k,x_d,x_l,x_u,g,dg,mov,exp)
    elif sub == 3:
        [x_p,x_d,dx_l,dx_u]=sub_oc_qpl(n,m,x_k,x_d,x_l,x_u,g,dg,mov,exp)
    elif sub == 10:
        [x_p,x_d,dx_l,dx_u]=sub_mma(n,m,x_k,x_d,x_l,x_u,g,dg,mov,asy)
    elif sub == 11:
        [x_p,x_d,dx_l,dx_u,L_k,U_k]=sub_mmaa(n,m,x_k,x_d,x_l,x_u,g,dg,x_1,x_2,L_k,U_k,k,mov,asy)
    elif sub == 12:
        [x_p,x_d,dx_l,dx_u]=sub_mma_rel(n,m,x_k,x_d,x_l,x_u,g,dg,mov,asy)
    elif sub == 13:
        [x_p,x_d,dx_l,dx_u,L_k,U_k]=sub_mma_rela(n,m,x_k,x_d,x_l,x_u,g,dg,x_1,x_2,L_k,U_k,k,mov,asy)
    elif sub == 20:
        [x_p,x_d,dx_l,dx_u]=sub_con(n,m,x_k,x_d,x_l,x_u,g,dg,mov)
    elif sub == 21:
        [x_p,x_d,dx_l,dx_u]=sub_con_exp(n,m,x_k,x_d,x_l,x_u,g,dg,x_1,dg_1,mov,exp)
    elif sub == 30:
        [x_p,x_d,dx_l,dx_u]=sub_qpq_exp(n,m,x_k,x_d,x_l,x_u,g,dg,x_1,dg_1,mov,exp)
    elif sub == 31:
        [x_p,x_d,dx_l,dx_u]=sub_qpl_exp(n,m,x_k,x_d,x_l,x_u,g,dg,x_1,dg_1,mov,exp)
    elif sub == 32:
        [x_p,x_d,dx_l,dx_u]=sub_qpl_exp_ne(n,m,x_k,x_d,x_l,x_u,g,dg,x_1,dg_1,mov,exp)
    elif sub == 33:
        [x_p,x_d,dx_l,dx_u]=sub_qpl_exp_ne_aml(n,m,x_k,x_d,x_l,x_u,g,dg,x_1,x_2,dg_1,mov,exp)
    elif sub == 99:
        [x_p,x_d,dx_l,dx_u,L_k,U_k]=sub_usr(n,m,x_k,x_d,x_l,x_u,g,dg,x_1,x_2,L_k,U_k,k,mov,asy,aux)
    else:
        print('ERROR; subsolver not found')
        stop
#
    return [x_p,x_d,dx_l,dx_u, L_k, U_k]
#
