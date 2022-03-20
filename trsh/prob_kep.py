#
import numpy as np
from scipy.optimize import minimize
#
from matplotlib import colors
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import cvxopt ;import cvxopt.cholmod
#
####################################################################################################
#
#   Simulation: Objective and constraints function value given x
#               derivatives if possible, else finite differences (f_d)
#
#   n   :   number of design variables
#   m   :   number of constraints
#   x_p :   current evaluation point (design variables)
#   aux :   auxilliary data passed from init (also available in subproblem)
#   glo :   sample number if multi-starts selected (else 0)
#   out :   flag used to activate output; called with flag = 1 after termination, else 0
#   
#   g   :   1D array of size m + 1, containing function value of objective (at index 0),
#           followed by constraint function values
#   dg  :   2D array of size (m+1,n), containing gradients of objective wrt design variables,
#           in row 0, followed by the gradients of constraints wrt design variables
#           (not used if finite differences are requested)
#
def simu(n,m,x_p,aux,glo,out):
#
    g=np.zeros((1+m),dtype=np.float64)
    dg=np.zeros((1+m,n),dtype=np.float64)
#
    [nelx,nely,volfrac,rmin,penal,ft,Emin,Emax,ndof,KE,H,Hs,iK,jK,edofMat,fixed,free,f,u,im,fig]=aux
#
    ce=np.zeros(n,dtype=np.float64)
    dc=np.zeros(n,dtype=np.float64)
    xPhys=np.zeros(n,dtype=np.float64)
    x=np.zeros(n,dtype=np.float64)
    x=x_p[:]
#
    # Filter design variables
    if ft==0:   xPhys[:]=x
    elif ft==1: xPhys[:]=np.asarray(H*x[np.newaxis].T/Hs)[:,0]
#
    f_tmp=np.zeros(ndof)
#
    np.add.at(f_tmp, edofMat[:, 1::2].flatten(),
                  np.kron(xPhys, -1e0 * np.ones(4) / 4))
#
    f[:,0] = f_tmp
#
    # Setup and solve FE problem
#
#   MATERIAL LAW AS PER KOPPEN
    sK=((KE.flatten()[np.newaxis]).T*(Emin+(0.1*xPhys+0.9*xPhys**penal)*(Emax-Emin))).flatten(order='F')
#
#   STANDARD MATERIAL LAW
#   sK=((KE.flatten()[np.newaxis]).T*(Emin+(xPhys)**penal*(Emax-Emin))).flatten(order='F')
#
    K = coo_matrix((sK,(iK,jK)),shape=(ndof,ndof)).tocsc()
    # Remove constrained dofs from matrix and convert to coo
    K = deleterowcol(K,fixed,fixed).tocoo()
    # Solve system 
    K = cvxopt.spmatrix(K.data,K.row.astype(np.int),K.col.astype(np.int))
    B = cvxopt.matrix(f[free,0])
    cvxopt.cholmod.linsolve(K,B)
    u[free,0]=np.array(B)[:,0]

    # Objective and sensitivity
    ce[:] = (np.dot(u[edofMat].reshape(nelx*nely,8),KE) * u[edofMat].reshape(nelx*nely,8) ).sum(1)
#
#
#   MATERIAL LAW AS PER KOPPEN
    obj=( (Emin + (0.1*xPhys + 0.9*xPhys**penal)*(Emax-Emin))*ce ).sum()
#
#   STANDARD MATERIAL LAW
#   obj=( (Emin+xPhys**penal*(Emax-Emin))*ce ).sum()
#
#   MATERIAL LAW AS PER KOPPEN
    dc[:]=-1e0*(Emax-Emin)*(0.1 + 0.9*penal*xPhys**(penal-1))*ce
#
#   STANDARD MATERIAL LAW
#   dc[:]=(-penal*xPhys**(penal-1)*(Emax-Emin))*ce
#
#
    dc[:] -= u[edofMat[:, 1], 0] * 1e0 / 2
    dc[:] -= u[edofMat[:, 3], 0] * 1e0 / 2
    dc[:] -= u[edofMat[:, 5], 0] * 1e0 / 2
    dc[:] -= u[edofMat[:, 7], 0] * 1e0 / 2
#
    g[0]=obj
#
    dv=-np.ones(nely*nelx,dtype=np.float64)/float(n)/volfrac
#   g[1]=volfrac*float(n)-np.sum(x)
    g[1]=1e0-np.sum(x)/float(n)/volfrac
#   g[1]=np.sum(x)-volfrac*float(n)
#
    # Sensitivity filtering:
    if ft==0:
        dg[0][:] = np.asarray((H*(x*dc))[np.newaxis].T/Hs)[:,0] / np.maximum(0.001,x)
        dg[1][:] = dv
    elif ft==1:
        dg[0][:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:,0]
        dg[1][:] = np.asarray(H*(dv[np.newaxis].T/Hs))[:,0]
#
    if out == 1:
        im.set_array(np.append(np.flip(-xPhys.reshape((nelx,nely)).T,axis=1),
            -xPhys.reshape((nelx,nely)).T,axis=1))
        fig.canvas.draw()
        plt.savefig('topo_%d.png'%glo)
#
    g=g/6.2e7
    dg[0]=dg[0]/6.2e7
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
#   aux :   auxiliary dictionary / list which may be used for custom parameters 
#           (passed to simu, and the subsolver)
#   glo :   Number of samples in multi-start run
#
#   Subproblem flags (sub)
#
#   1   :   OC (restricted to min. comp. problem)
#   2   :   generalised OC (restricted to min. comp. problem)
#   10  :   MMA
#   11  :   MMA with asymptote adaptation heuristic (as per Svanberg 1987) 
#   12  :   Same as 10, but with constraint relaxation (as per Svanverg 1987)
#   13  :   Same as 11, but with constraint relaxation (as per Svanberg 1987)
#   20  :   CONLIN
#   21  :   CONLIN with adaptive exponent
#   30  :   QCQP reciprocal adaptive
#   31  :   QPLP reciprocal adaptive
#   100 :   OSQP QP reciprocal adaptive
#   101 :   LP; solved with OSQP with exact zero Hessian
#   102 :   OSQP QP spherical quadratic approximation
#   999 :   LP with AML; solved with OSQP with exact zero Hessian
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
#   c_e=*   :   convergence threshold in terms of Euclidean norm of step size
#   c_i=*   :   convergence threshold in terms of Infinity norm of step size
#   c_v=*   :   threshold for constraint violation (else no termination)
#   f_t=*   :   threshold for termination on objective value change (not recommended)
#   f_a=*   :   a priori known (analytic) solution; used as termination criteria in some tests
#               (set to large negative number otherwise, as it is not recommended)
#   m_k     :   maximum number of iterations
#
def init():
#
    n=100*100
    m=1
#
    x_i=0.2*np.ones(n,dtype=np.float64)
    x_l=0e0*np.ones(n,dtype=np.float64)
    x_u=1e0*np.ones(n,dtype=np.float64)
#
    f_d=0
    c_e=1e-1
    c_i=1e-6
    c_v=1e-1
    f_t=0e0
    f_a=-1e8
    m_k=99
#
    sub=100#999
    glo=0
    cpu=0
#
    mov_abs=0.2e0
    mov_rel=2e0
    mov_fct=0.5e0*np.ones(n,dtype=np.float64)
#
    asy_fac=1e0/2e0
    asy_adp=1e0/2e0
#
    exp_set=1e0
    exp_min=-6e0 
    exp_max=-1e0
#
    mov={'mov_abs': mov_abs, 'mov_rel': mov_rel, 'mov_fct': mov_fct}
    exp={'exp_set': exp_set, 'exp_min': exp_min, 'exp_max': exp_max}
    asy={'asy_fac': asy_fac,'asy_adp': asy_adp}
#
    aux=[]
#
    aux=topopt_init(100,100,0.2,5.4,3.0,1)
#
    return n,m,x_i,x_l,x_u,c_e,c_i,c_v,f_t,f_a,m_k,f_d,sub,mov,asy,exp,aux,glo,cpu
#
def topopt_init(nelx,nely,volfrac,rmin,penal,ft):
#
    # Max and min stiffness
    Emin=1e-9
    Emax=1.0

    # dofs:
    ndof = 2*(nelx+1)*(nely+1)

    # Allocate design variables (as array), initialize and allocate sens.
    x=volfrac * np.ones(nely*nelx,dtype=float)
    xold=x.copy()
    xPhys=x.copy()

    g=0 # must be initialized to use the NGuyen/Paulino OC approach
    dc=np.zeros((nely,nelx), dtype=float)

	# FE: Build the index vectors for the for coo matrix format.
    KE=lk()
    edofMat=np.zeros((nelx*nely,8),dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely+elx*nely
            n1=(nely+1)*elx+ely
            n2=(nely+1)*(elx+1)+ely
            edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])
    # Construct the index pointers for the coo format
    iK = np.kron(edofMat,np.ones((8,1))).flatten()
    jK = np.kron(edofMat,np.ones((1,8))).flatten()    

    # Filter: Build (and assemble) the index+data vectors for the coo matrix format
    nfilter=int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2))
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc=0
    for i in range(nelx):
        for j in range(nely):
            row=i*nely+j
            kk1=int(np.maximum(i-(np.ceil(rmin)-1),0))
            kk2=int(np.minimum(i+np.ceil(rmin),nelx))
            ll1=int(np.maximum(j-(np.ceil(rmin)-1),0))
            ll2=int(np.minimum(j+np.ceil(rmin),nely))
            for k in range(kk1,kk2):
                for l in range(ll1,ll2):
                    col=k*nely+l
                    fac=rmin-np.sqrt(((i-k)*(i-k)+(j-l)*(j-l)))
                    iH[cc]=row
                    jH[cc]=col
                    sH[cc]=np.maximum(0.0,fac)
                    cc=cc+1
    # Finalize assembly and convert to csc format
    H=coo_matrix((sH,(iH,jH)),shape=(nelx*nely,nelx*nely)).tocsc()	
    Hs=H.sum(1)

    # BC's and support
    ndofy=2 * (nely + 1)
    dofs=np.arange(2*(nelx+1)*(nely+1))
    fixed = np.union1d(dofs[0:ndofy:2],
        np.array([ndof - 1]))
#   fixed=np.union1d(dofs[0:2*(nely+1):2],np.array([2*(nelx+1)*(nely+1)-1]))
    free=np.setdiff1d(dofs,fixed)

    # Solution and RHS vectors
    f=np.zeros((ndof,1))
    u=np.zeros((ndof,1))
 
    # Initialize plot and plot the initial design
    plt.ion() # Ensure that redrawing is possible
    fig,ax = plt.subplots()
    im = ax.imshow(np.append(np.flip(-xPhys.reshape((nelx,nely)).T),
        -xPhys.reshape((nelx,nely)).T,axis=1), cmap='gray',
        interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
#
    return nelx,nely,volfrac,rmin,penal,ft,Emin,Emax,ndof,KE,H,Hs,iK,jK,edofMat,fixed,free,f,u,im,fig
#
#element stiffness matrix
def lk():
	E=1
	nu=0.3
	k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
	KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
	[k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
	[k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
	[k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
	[k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
	[k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
	[k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
	[k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ]);
	return (KE)
#
def deleterowcol(A, delrow, delcol):
    # Assumes that matrix is in symmetric csc form !
    m = A.shape[0]
    keep = np.delete (np.arange(0, m), delrow)
    A = A[keep, :]
    keep = np.delete (np.arange(0, m), delcol)
    A = A[:, keep]
    return A
#
