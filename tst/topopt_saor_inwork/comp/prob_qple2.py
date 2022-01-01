#
import numpy as np
from scipy.optimize import minimize
import utils
import matplotlib.pyplot as plt
from matplotlib import colors
#
####################################################################################################
#
#   Simulation: Objective and constraints function value given x
#               derivatives if possible, else finite differences (f_d)
#
#   n   :   number of design variables
#   m   :   number of constraints
#   x_p :   current evaluation point (design variables)
#   
#   g   :   1D array of size m + 1, containing function value of objective (at index 0),
#           followed by constraint function values
#   dg  :   2D array of size (m+1,n), containing gradients of objective wrt design variables,
#           in row 0, followed by the gradients of constraints wrt design variables
#           (not used if finite differences are requested)
#
def simu(n,m,x_p,aux,out):
#
#   from init
#
    nx=50*2
    ny=25*2
    eps = 1e-6
    mesh = utils.Mesh(nx, ny)
    factor = None
    fradius = 2
    penal=3
    vf=0.2
    x0 = vf * np.ones(mesh.n, dtype=float)

    dc = np.zeros((mesh.nely, mesh.nelx), dtype=float)
    ce = np.ones(mesh.n, dtype=float)

    ke = utils.element_matrix_stiffness()

    filt = utils.Filter(mesh, fradius)

    dofs = np.arange(mesh.ndof)
    fixed = np.union1d(dofs[0:2 * (mesh.nely + 1):2],
                                np.array([mesh.ndof - 1]))
    free = np.setdiff1d(dofs, fixed)
    f = np.zeros(mesh.ndof, dtype=float)
    u = np.zeros((mesh.ndof, 1), dtype=float)

    dout = 1
    f[dout] = -1
#
    g=np.zeros((1+m),dtype=np.float64)

    xphys = filt.forward(x_p)

    ym = eps + (xphys.flatten() ** penal) * (1 - eps)
    stiffness_matrix = utils.assemble_K(ym, mesh, fixed)

    u[free, :] = utils.linear_solve(stiffness_matrix, f[free])

    ce[:] = (np.dot(u[mesh.edofMat].reshape(mesh.n, 8), ke) * \
        u[mesh.edofMat].reshape(mesh.n, 8)).sum(1)

    g[0] = np.dot(f, u)
    g[1] = np.sum(xphys[:]) / (vf * mesh.n) - 1
#
    dg=np.zeros((1+m,n),dtype=np.float64)
#
    xphys = filt.forward(x_p)
    dg[0, :] -= (1 - eps) * (penal * xphys ** (penal - 1)) * ce
    dg[1, :] = np.ones(mesh.n) / (vf * mesh.n)
    dg[0, :] = filt.backward(dg[0, :])
    dg[1, :] = filt.backward(dg[1, :])
#
    if out == 1:
        figdes, axsdes = plt.subplots(1)
        axsdes.imshow(-xphys.reshape((mesh.nelx, mesh.nely)).T, cmap='gray',
                         interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
        plt.savefig('topo.png')
#
    return [g,dg]
#
####################################################################################################
#
#   Initialisation: Set problem size and starting point
#
#   n   :   number of design variables
#   m   :   number of constraints
#   x_i :   starting point
#   x_l :   global lower bounds on design variables
#   x_u :   global upper bounds on design variables
#   aux :   auxiliary dictionary which may be used for custom parameters (passing of)
#
#   Subproblem flags (sub)
#
#   10  :   MMA
#   11  :   MMA with asymptote adaptation heuristic (as per Svanberg 1987) 
#   12  :   Same as 10, but with constraint relaxation (as per Svanverg 1987)
#   13  :   Same as 11, but with constraint relaxation (as per Svanberg 1987)
#   20  :   CONLIN
#   21  :   CONLIN with adaptive exponent
#   30  :   QCQP reciprocal adaptive
#   31  :   QPLP reciprocal adaptive
#
#   Suproblem parameters
#
#   10,12   :   asy_fac     :       Simple rule from Svanberg 1987 ('t')
#           :   mov_rel     :       Svanberg 1987 bounds factor (24)
#
#   11,13   :   asy_adp     :       L and U heuristic Svanberg 1987 ('s')
#           :   mov_rel     :       Svanberg 1987 bounds factor (24)
#
#   20      :   mov_abs     :       Move-limit as factor of design domain
#           :   mov_rel     :       if mov_abs is set to < 0, then Svanberg 
#                                   bounds factor is used
#   21      :   mov_*       :       Same as 20
#           :   exp_set     :       set constant exponent (e.g. -2 instead of -1)
#           :   exp_min     :       if exp_set is set to > 0, then adaptive exponential
#                                   fitting is used, requiring ...
#           :   exp_max     :       maximum exponent bound (-0.1 suggested)
#           :   exp_min     :       minimum exponent bound (-6 suggested)
#
#   30,31   :   mov_*       :       Same as 20
#           :   exp_*       :       Same as 21
#
#   Global flags and parameters
#
#   f_d=1   :   activate gradient calculation via finite differences (dont if 0)
#   c_t=*   :   convergence limit in terms of step size (Euclidean norm)
#   f_a=*   :   a priori known (analytic) solution; used in termination criteria (set to large 
#               negative number if not applicable)
#   m_k     :   maximum outer iterations
#
def init():
#
    n=1250*4
    m=1
#
    x_i=.2*np.ones(n,dtype=np.float64)
    x_l=1e-3*np.ones(n,dtype=np.float64)
    x_u=1e0*np.ones(n,dtype=np.float64)
#
    f_d=0
    c_t=1e-1
    f_a=-1e8
    m_k=499
#
    sub=32  #!
    glo=0
#
    mov_abs=0.2e0 #!
    mov_rel=2e0
#
    asy_fac=1e0/2e0
    asy_adp=1e0/2e0
#
    exp_set=2e0
    exp_min=-4e0 #!
    exp_max=1e0
#
    mov={'mov_abs': mov_abs, 'mov_rel': mov_rel}
    exp={'exp_set': exp_set, 'exp_min': exp_min, 'exp_max': exp_max}
    asy={'asy_fac': asy_fac,'asy_adp': asy_adp}
    aux={}
#
    return n,m,x_i,x_l,x_u,c_t,f_a,m_k,f_d,sub,mov,asy,exp,aux,glo
#

