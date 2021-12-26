#
import numpy as np  
from sub_mma import sub_mma
from sub_mma_rel import sub_mma_rel
from sub_con import sub_con
#
def subs(sub, n, m, x_k, x_d, x_l, x_u, g, dg, mov, mov_rel, asy_fac):
#
    if sub == 1:
        [x_p,x_d,dx_l,dx_u]=sub_mma(n, m, x_k, x_d, x_l, x_u, g, dg, mov, mov_rel, asy_fac)
    elif sub == 2:
        [x_p,x_d,dx_l,dx_u]=sub_mma_rel(n, m, x_k, x_d, x_l, x_u, g, dg, mov, mov_rel, asy_fac)
    elif sub == 3:
        [x_p,x_d,dx_l,dx_u]=sub_con(n, m, x_k, x_d, x_l, x_u, g, dg, mov, mov_rel)
    else:
        print('ERROR; subsolver not found')
        stop
#
    return [x_p,x_d,dx_l,dx_u]
#
