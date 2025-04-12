import sys
import os
base_dir = os.path.dirname(os.path.dirname(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
from imports import *
from qLPV_model import qLPV_model
import cvxpy as cp

jax.config.update("jax_enable_x64", True)


#Load model
current_directory = Path(__file__).parent.parent
file_path = current_directory / "identification_results/oscillator/dataset.pkl"
with open(file_path, 'rb') as f:
    dataset = pickle.load(f)

file_path = current_directory / "identification_results/oscillator/new_0.07.pkl"
with open(file_path, 'rb') as f:
    models = pickle.load(f)

#Extract system
system = dataset['system']
model_LPV_concur = models['model_LPV_concur']
RCI_concur = models['RCI_concur']
dt = system.dt if hasattr(system, 'dt') else 1.0

A = model_LPV_concur['A'].copy()
B = model_LPV_concur['B'].copy()
C = model_LPV_concur['C'].copy()
L = model_LPV_concur['L'].copy()
Win = model_LPV_concur['Win'].copy()
bin = model_LPV_concur['bin'].copy()
Whid = model_LPV_concur['Whid'].copy()
bhid = model_LPV_concur['bhid'].copy()
Wout = model_LPV_concur['Wout'].copy()
bout = model_LPV_concur['bout'].copy()
nx = A.shape[1]
nu = B.shape[2]
ny = C.shape[0]

F = RCI_concur['F']
E = RCI_concur['E']
V = RCI_concur['V']
m = F.shape[0]
m_bar = len(V)
HY = dataset['constraints']['HY']
hY = dataset['constraints']['hY']
mY = HY.shape[0]
Y_set = Polytope(A=HY,b=hY)
Y_vert = Y_set.V
vY = len(Y_vert)
HU = dataset['constraints']['HU']
hU = dataset['constraints']['hU']
mU = HU.shape[0]
Ys_observer = dataset['Ys_observer']
Us_observer = dataset['Us_observer']
Zs_observer = np.hstack((Us_observer, Ys_observer))
nq = Win.shape[0]
nH = 0
activation = models['activation']
N_MPC = models['N_MPC']
kappa = models['kappa']
regularization = 0.001
tolerances = 1e-4
param_idx_plt = 4
zetas = [0.07]
N_iters = 200

#Check BFRs
from utils.qLPV_BFR import qLPV_BFR
Us_train = dataset['Us_train']
Ys_train = dataset['Ys_train']
Us_observer = dataset['Us_observer']
Ys_observer = dataset['Ys_observer']
Us_test = dataset['Us_test']
Ys_test = dataset['Ys_test']
Us_train_comp = Us_train[0]
Ys_train_comp = Ys_train[0]
if len(Us_train) > 1:
    for i in range(1,len(Us_train)):
        Us_train_comp = np.vstack((Us_train_comp, Us_train[i]))
        Ys_train_comp = np.vstack((Ys_train_comp, Ys_train[i]))
model_LPV_BFR = {'A': A, 'B': B, 'L': L, 'C': C, 'Win': Win, 'bin': bin, 'Whid': Whid, 'bhid': bhid, 'Wout': Wout, 'bout': bout}
BFR_train_qLPV_OL, y_train_qLPV_OL, x0_train_qLPV_OL = qLPV_BFR(model_LPV_BFR, [Us_train_comp], [Ys_train_comp], observer = False, activation_func = activation)
print('BFR train: qLPV OL', BFR_train_qLPV_OL)

BFR_observer_qLPV_OL, y_observer_qLPV_OL, x0_observer_qLPV_OL = qLPV_BFR(model_LPV_BFR, [Us_observer], [Ys_observer], observer = False, activation_func = activation)
print('BFR observer: qLPV OL', BFR_observer_qLPV_OL)

BFR_test_qLPV_OL, y_test_qLPV_OL, x0_test_qLPV_OL = qLPV_BFR(model_LPV_BFR, [Us_test], [Ys_test], observer = False, activation_func = activation)
print('BFR test: qLPV OL', BFR_test_qLPV_OL)


if activation == 1:
    activation = nn.swish
elif activation == 2:
    activation = nn.relu
elif activation == 3:
    @jax.jit
    def activation(x):
        return nn.elu(x)+1.
    
@jax.jit
def parameter_fcn(x,u):
    z = x
    p = jnp.zeros((nq,))
    for i in range(nq):
        post_linear = Win[i]@z+bin[i]
        post_activation = activation(post_linear)
        for j in range(nH-1): 
            post_linear = Whid[i][j]@post_activation+bhid[i][j]
            post_activation = activation(post_linear)
        post_linear = Wout[i]@post_activation+bout[i]
        p = p.at[i].set(post_linear)
    p = jnp.exp(p)/jnp.sum(jnp.exp(p))
    return p

@jax.jit
def state_fcn_obsv(x,z):
    p = parameter_fcn(x,z[0:nu])
    A_tot = jnp.zeros((nx,nx))  
    B_tot = jnp.zeros((nx,nu))
    L_tot = jnp.zeros((nx,ny))
    for i in range(nq):
        A_tot += A[i]*p[i]
        B_tot += B[i]*p[i]
        L_tot += L[i]*p[i]
    return (A_tot-L_tot@C)@x+jnp.hstack((B_tot,L_tot))@z

@jax.jit
def output_fcn(x,u):
    return C@x

@jax.jit
def SS_forward_output(x, z):
    #This is on the observer
    y_next = output_fcn(x, z)
    x_next = state_fcn_obsv(x, z)
    return x_next, jnp.hstack((x_next,y_next))

@jax.jit
def bound_propagation(yRCI_comp, zeta_use):
    RCI_vertices = jnp.zeros((0,nx))
    for i in range(m_bar):
        RCI_vertices = jnp.vstack((RCI_vertices, V[i]@yRCI_comp))

    max_RCI_vertices = jnp.max(RCI_vertices,axis=0)
    min_RCI_vertices = jnp.min(RCI_vertices,axis=0)
    mean_input = 0.5*(max_RCI_vertices+min_RCI_vertices)
    var_input = 0.5*(max_RCI_vertices-min_RCI_vertices)+zeta_use
            
    lu_before = jnp.zeros((nq, 2))
    for i in range(nq):
        lb_curr = Win[i]@mean_input+bin[i]-jnp.abs(Win[i])@var_input
        ub_curr = Win[i]@mean_input+bin[i]+jnp.abs(Win[i])@var_input
        lb_activated = activation(lb_curr)
        ub_activated = activation(ub_curr)
        mean_lb_activated = 0.5*(lb_activated+ub_activated)
        var_lb_activated = 0.5*(ub_activated-lb_activated)
        for j in range(nH-1):     
            lb_curr = Whid[i][j]@mean_lb_activated+bhid[i][j]-jnp.abs(Whid[i][j])@var_lb_activated
            ub_curr = Whid[i][j]@mean_lb_activated+bhid[i][j]+jnp.abs(Whid[i][j])@var_lb_activated
            lb_activated = activation(lb_curr)
            ub_activated = activation(ub_curr)
            mean_lb_activated = 0.5*(lb_activated+ub_activated)
            var_lb_activated = 0.5*(ub_activated-lb_activated)
        lb_curr = Wout[i]@mean_lb_activated+bout[i]-jnp.abs(Wout[i])@var_lb_activated
        ub_curr = Wout[i]@mean_lb_activated+bout[i]+jnp.abs(Wout[i])@var_lb_activated
        lu_before = lu_before.at[i].set(jnp.hstack((jnp.exp(lb_curr), jnp.exp(ub_curr))))
    
    a_next = jnp.zeros((nq,))
    for i in range(nq):
        numerator = lu_before[i,0]
        denominator = lu_before[i,0]
        for j in range(nq):
            if i!=j:
                denominator = denominator+lu_before[j,1]
        a_next = a_next.at[i].set(numerator/denominator)

    return a_next, mean_input, var_input

H_incr = np.vstack((np.eye(nx),-np.eye(nx)))
H_incr_j = np.zeros((m_bar,2*nx,m))
for j in range(m_bar):
    H_incr_j[j] = H_incr@V[j]

simulator = partial(SS_forward_output)
x_next, xy_next = jax.lax.scan(simulator, jnp.zeros((nx,)), Zs_observer)
x_observer = xy_next[:,0:nx]
y_observer = xy_next[:,nx:nx+ny]


@jax.jit
def get_x_in_B(carry,x_sample,bound_lb,bound_ub):
    diff = np.vstack((np.eye(nx),-np.eye(nx)))@x_sample - jnp.hstack((bound_ub,-bound_lb))
    diff_max = jnp.max(diff,axis=0)
    inclusion = jnp.where(diff_max <= 0, 1.0, 0.0)
    return carry,jnp.array([inclusion])

target_y_vertices = np.zeros((0,))
for i in range(vY):
    target_y_vertices = np.hstack((target_y_vertices, np.tile(Y_vert[i], reps=(1,N_MPC))[0]))

@jax.jit
def RCI_computation_noBP():
    
    #Disturbance set
    error = Ys_observer-y_observer
    w_hat = (jnp.max(error,axis=0)+jnp.min(error,axis=0))/2
    eps_w = (jnp.max(error,axis=0)-jnp.min(error,axis=0))/2
    
    #Build QP
    A_mean = jnp.mean(A, axis=0)
    B_mean = jnp.mean(B, axis=0)

    #Dynamics
    M_matrix = jnp.zeros((nx*N_MPC,nu*(N_MPC)))
    for i in np.arange(N_MPC):
        for j in np.arange(i+1):
            id_loc = np.arange(j*nu,(j+1)*nu)
            A_prod = jnp.eye(nx)
            for k in np.arange(i-j):
                A_prod = A_prod@A_mean
            M_matrix = M_matrix.at[i*nx:(i+1)*nx,id_loc].set(A_prod@B_mean)

    A_ineq = jnp.tile(-jnp.eye(m),reps=(N_MPC*vY,1))
    A_ineq = jnp.hstack((A_ineq, jnp.zeros((m*N_MPC*vY,nu*m_bar))))
    F_M = jnp.kron(jnp.eye(N_MPC),F)@M_matrix
    A_ineq = jnp.hstack((A_ineq, jnp.kron(jnp.eye(vY),F_M)))
    b_ineq = jnp.zeros((m*N_MPC*vY,))
    A_ineq = jnp.vstack((A_ineq, 
                        jnp.hstack((jnp.zeros((mU*vY*N_MPC,m+nu*m_bar)),jnp.kron(jnp.eye(vY*N_MPC),HU)))))
    b_ineq = jnp.hstack((b_ineq, jnp.tile(hU, reps=(1,vY*N_MPC))[0]))

    U_selector = jnp.hstack((jnp.zeros((nu*vY*N_MPC,m+nu*m_bar)), jnp.eye(nu*vY*N_MPC)))
    X_builder = jnp.kron(jnp.eye(vY), M_matrix)
    Y_builder = jnp.kron(jnp.eye(vY*N_MPC), C)

    Z_total = Y_builder@X_builder@U_selector
    Q1 = 2*Z_total.T@Z_total
    c1 = -2*Z_total.T@target_y_vertices

    #Build constraints
    #RCI
    FA_part = jnp.zeros((m*m_bar*nq, m))
    FB_part = jnp.zeros((m*m_bar*nq, nu*m_bar))
    Fb_part = jnp.zeros((m*m_bar*nq, vY*N_MPC*nu))
    part_1 = jnp.zeros((m*m_bar*nq,))
    for i in range(nq):
        for k in range(m_bar):
            FA_part = FA_part.at[(i*m_bar+k)*m:(i*m_bar+k+1)*m].set(F@A[i]@V[k]-jnp.eye(m))
            FB_part = FB_part.at[(i*m_bar+k)*m:(i*m_bar+k+1)*m,k*nu:(k+1)*nu].set(F@B[i])
            part_1 = part_1.at[(i*m_bar+k)*m:(i*m_bar+k+1)*m].set(-(F@L[i]@w_hat+kappa*jnp.abs(F@L[i])@eps_w))
    
    #Output constraints
    HC_part = jnp.zeros((m_bar*mY, m))
    HD_part = jnp.zeros((m_bar*mY, nu*m_bar))
    Hb_part = jnp.zeros((m_bar*mY, vY*N_MPC*nu))
    part_2 = jnp.zeros((m_bar*mY,))
    for i in range(m_bar):
        HC_part = HC_part.at[i*mY:(i+1)*mY].set(HY@C@V[i])
        part_2 = part_2.at[i*mY:(i+1)*mY].set(hY - (HY@w_hat+kappa*jnp.abs(HY)@eps_w))

    #Input constraints
    U1_part = jnp.zeros((m_bar*mU, m))
    U2_part = jnp.zeros((m_bar*mU, nu*m_bar))
    Ub_part = jnp.zeros((m_bar*mU, vY*N_MPC*nu))
    part_3 = jnp.zeros((m_bar*mU,))
    for i in range(m_bar):
        U2_part = U2_part.at[i*mU:(i+1)*mU,i*nu:(i+1)*nu].set(HU)
        part_3 = part_3.at[i*mU:(i+1)*mU].set(hU)

    #Config cons
    size_E = E.shape[0]
    CC_1 = E
    CC_2 = jnp.zeros((size_E, nu*m_bar))
    CC_3 = jnp.zeros((size_E, vY*N_MPC*nu))
    part_5 = jnp.zeros((size_E,))

    A_ineq = jnp.vstack((A_ineq,
                        jnp.hstack((FA_part, FB_part, Fb_part)),
                        jnp.hstack((HC_part, HD_part, Hb_part)),
                        jnp.hstack((U1_part, U2_part, Ub_part)),
                        jnp.hstack((CC_1, CC_2, CC_3))))
    b_ineq = jnp.hstack((b_ineq, part_1, part_2, part_3,part_5))

    num_vars = m+nu*m_bar+vY*N_MPC*nu
    Q = Q1+regularization*np.eye(num_vars)
    c = c1
    A_eq = jnp.zeros((0,num_vars))
    b_eq = jnp.zeros((0,))

    QP_soln = qpax.solve_qp_primal(Q, c, A_eq, b_eq, A_ineq, b_ineq, solver_tol=tolerances, target_kappa=tolerances)
    
    yRCI_comp = QP_soln[0:m]
    uRCI_comp = QP_soln[m:m+nu*m_bar].reshape(m_bar,nu)
    y_trajec = Z_total@QP_soln
    error = target_y_vertices - y_trajec
    cost = jnp.dot(error,error)

    return cost, yRCI_comp, uRCI_comp, eps_w, w_hat

cost_noBP, yRCI_noBP, uRCI_noBP, eps_w_noBP, w_hat_noBP = RCI_computation_noBP()
print('cost_noBP:', cost_noBP)

@jax.jit
def RCI_computation(yRCI_prev, zeta_use, factor = 1):
    a_used, mean_input, var_input = bound_propagation(yRCI_prev, zeta_use)
    a_used = a_used*factor
    h_incr = jnp.hstack((var_input,var_input))+H_incr@mean_input

    bound_lb = mean_input-var_input
    bound_ub = mean_input+var_input
    
    #Disturbance set
    error = Ys_observer-y_observer
    w_hat = (jnp.max(error,axis=0)+jnp.min(error,axis=0))/2
    eps_w = (jnp.max(error,axis=0)-jnp.min(error,axis=0))/2
    
    
    A_vertices = jnp.zeros((nq,nx,nx))
    B_vertices = jnp.zeros((nq,nx,nu))
    L_vertices = jnp.zeros((nq,nx,ny))
    for i in range(nq):
        A_loc = (1-jnp.sum(a_used))*A[i]
        B_loc = (1-jnp.sum(a_used))*B[i]
        L_loc = (1-jnp.sum(a_used))*L[i]
        for j in range(nq):
            A_loc = A_loc + a_used[j]*A[j]
            B_loc = B_loc + a_used[j]*B[j]
            L_loc = L_loc + a_used[j]*L[j]
        A_vertices = A_vertices.at[i].set(A_loc)
        B_vertices = B_vertices.at[i].set(B_loc)
        L_vertices = L_vertices.at[i].set(L_loc)

    #Build QP
    A_mean = jnp.mean(A_vertices, axis=0)
    B_mean = jnp.mean(B_vertices, axis=0)

    #Dynamics
    M_matrix = jnp.zeros((nx*N_MPC,nu*(N_MPC)))
    for i in np.arange(N_MPC):
        for j in np.arange(i+1):
            id_loc = np.arange(j*nu,(j+1)*nu)
            A_prod = jnp.eye(nx)
            for k in np.arange(i-j):
                A_prod = A_prod@A_mean
            M_matrix = M_matrix.at[i*nx:(i+1)*nx,id_loc].set(A_prod@B_mean)

    A_ineq = jnp.tile(-jnp.eye(m),reps=(N_MPC*vY,1))
    A_ineq = jnp.hstack((A_ineq, jnp.zeros((m*N_MPC*vY,nu*m_bar))))
    F_M = jnp.kron(jnp.eye(N_MPC),F)@M_matrix
    A_ineq = jnp.hstack((A_ineq, jnp.kron(jnp.eye(vY),F_M)))
    b_ineq = jnp.zeros((m*N_MPC*vY,))
    A_ineq = jnp.vstack((A_ineq, 
                        jnp.hstack((jnp.zeros((mU*vY*N_MPC,m+nu*m_bar)),jnp.kron(jnp.eye(vY*N_MPC),HU)))))
    b_ineq = jnp.hstack((b_ineq, jnp.tile(hU, reps=(1,vY*N_MPC))[0]))

    U_selector = jnp.hstack((jnp.zeros((nu*vY*N_MPC,m+nu*m_bar)), jnp.eye(nu*vY*N_MPC)))
    X_builder = jnp.kron(jnp.eye(vY), M_matrix)
    Y_builder = jnp.kron(jnp.eye(vY*N_MPC), C)

    Z_total = Y_builder@X_builder@U_selector
    Q1 = 2*Z_total.T@Z_total
    c1 = -2*Z_total.T@target_y_vertices

    #Build constraints
    #RCI
    FA_part = jnp.zeros((m*m_bar*nq, m))
    FB_part = jnp.zeros((m*m_bar*nq, nu*m_bar))
    Fb_part = jnp.zeros((m*m_bar*nq, vY*N_MPC*nu))
    part_1 = jnp.zeros((m*m_bar*nq,))
    for i in range(nq):
        for k in range(m_bar):
            FA_part = FA_part.at[(i*m_bar+k)*m:(i*m_bar+k+1)*m].set(F@A_vertices[i]@V[k]-jnp.eye(m))
            FB_part = FB_part.at[(i*m_bar+k)*m:(i*m_bar+k+1)*m,k*nu:(k+1)*nu].set(F@B_vertices[i])
            part_1 = part_1.at[(i*m_bar+k)*m:(i*m_bar+k+1)*m].set(-(F@L_vertices[i]@w_hat+kappa*jnp.abs(F@L_vertices[i])@eps_w))
    
    #Output constraints
    HC_part = jnp.zeros((m_bar*mY, m))
    HD_part = jnp.zeros((m_bar*mY, nu*m_bar))
    Hb_part = jnp.zeros((m_bar*mY, vY*N_MPC*nu))
    part_2 = jnp.zeros((m_bar*mY,))
    for i in range(m_bar):
        HC_part = HC_part.at[i*mY:(i+1)*mY].set(HY@C@V[i])
        part_2 = part_2.at[i*mY:(i+1)*mY].set(hY - (HY@w_hat+kappa*jnp.abs(HY)@eps_w))

    #Input constraints
    U1_part = jnp.zeros((m_bar*mU, m))
    U2_part = jnp.zeros((m_bar*mU, nu*m_bar))
    Ub_part = jnp.zeros((m_bar*mU, vY*N_MPC*nu))
    part_3 = jnp.zeros((m_bar*mU,))
    for i in range(m_bar):
        U2_part = U2_part.at[i*mU:(i+1)*mU,i*nu:(i+1)*nu].set(HU)
        part_3 = part_3.at[i*mU:(i+1)*mU].set(hU)

    #Config cons
    size_E = E.shape[0]
    CC_1 = E
    CC_2 = jnp.zeros((size_E, nu*m_bar))
    CC_3 = jnp.zeros((size_E, vY*N_MPC*nu))
    part_5 = jnp.zeros((size_E,))

    #Increment constraints
    incr_1 = jnp.zeros((2*m_bar*nx,m))
    incr_2 = jnp.zeros((2*m_bar*nx,nu*m_bar))
    incr_3 = jnp.zeros((2*m_bar*nx,vY*N_MPC*nu))
    part_6 = jnp.zeros((2*m_bar*nx,))
    for j in range(m_bar):
        incr_1 = incr_1.at[j*2*nx:(j+1)*2*nx].set(H_incr_j[j,:,:m])
        part_6 = part_6.at[j*2*nx:(j+1)*2*nx].set(h_incr)

    A_ineq = jnp.vstack((A_ineq,
                        jnp.hstack((FA_part, FB_part, Fb_part)),
                        jnp.hstack((HC_part, HD_part, Hb_part)),
                        jnp.hstack((U1_part, U2_part, Ub_part)),
                        jnp.hstack((CC_1, CC_2, CC_3)),
                        jnp.hstack((incr_1, incr_2, incr_3))))

    b_ineq = jnp.hstack((b_ineq, part_1, part_2, part_3,part_5,part_6))

    num_vars = m+nu*m_bar+vY*N_MPC*nu
    Q = Q1+regularization*np.eye(num_vars)
    c = c1
    A_eq = jnp.zeros((0,num_vars))
    b_eq = jnp.zeros((0,))

    QP_soln = qpax.solve_qp_primal(Q, c, A_eq, b_eq, A_ineq, b_ineq, solver_tol=tolerances, target_kappa=tolerances)
    
    yRCI_comp = QP_soln[0:m]
    uRCI_comp = QP_soln[m:m+nu*m_bar].reshape(m_bar,nu)
    y_trajec = Z_total@QP_soln
    error = target_y_vertices - y_trajec
    cost = jnp.dot(error,error)

    a_new, _, _ = bound_propagation(yRCI_comp,zeta_use)

    return cost, yRCI_comp, uRCI_comp, eps_w, w_hat, h_incr

cost_all = np.zeros((len(zetas),N_iters))
for qqq in range(len(zetas)):
    print('zeta=',zetas[qqq])
    zeta = zetas[qqq]
    _, yRCI_init, uRCI_init, _, _, _ = RCI_computation(100*np.ones((m,)), zeta, factor = 1)
    a_iters = np.zeros((1,nq))
    cost_iters = np.zeros((0,))
    y_iters = np.float64([yRCI_init])
    eps_w_iters = np.zeros((0,))
    w_hat_iters = np.zeros((0,))
    h_incr_iters = np.zeros((0,))

    for i in range(N_iters):
        cost, yRCI_comp, uRCI_comp, eps_w_comp, w_hat_comp, h_incr_comp = RCI_computation(y_iters[-1], zeta)
        a_new, _, _ = bound_propagation(yRCI_comp, zeta)
        a_iters = np.vstack((a_iters, np.float64(a_new)))
        cost_iters = np.hstack((cost_iters, cost))
        y_iters = np.vstack((y_iters, np.float64(yRCI_comp)))
        eps_w_iters = np.hstack((eps_w_iters, eps_w_comp))
        w_hat_iters = np.hstack((w_hat_iters, w_hat_comp))
        h_incr_iters = np.hstack((h_incr_iters, h_incr_comp))
    print('Cost final = ', np.min(cost_iters))
    print('a final = ', a_iters[-1])

    cost_all[qqq] = cost_iters

import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": True,     # Use LaTeX for all text rendering
    "font.family": "serif",  # Use a serif font
})
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
fig, ax = plt.subplots()
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
colors = plt.cm.magma(np.linspace(0, 1, len(zetas)))
for i in range(len(zetas)):
    ax.plot(cost_all[i], 
            label=rf'$\zeta = {zetas[i]}$', 
            color=colors[i])
ax.set_xscale('log')
# ax.set_yscale('log')
ax.legend()
ax.grid(True)
ax.set_xlim(1, N_iters)  # Note: use 1 as lower bound for log-scale x-axis.
plt.tight_layout()

# Define a formatter that wraps the tick label in LaTeX math mode.
formatter = FuncFormatter(lambda x, pos: r'${:g}$'.format(x))
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)  # Apply to y-axis if desired.
plt.show()

fig, ax = plt.subplots()
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
x_values = np.arange(191, 200)
for i in range(len(zetas)):
    ax.plot(x_values, cost_all[i,191:200], 
            color=colors[i])
ax.grid(True)
ax.set_xlim(191, 200) 
plt.tight_layout()

formatter = FuncFormatter(lambda x, pos: r'${:g}$'.format(x))
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)  # Apply to y-axis if desired.

plt.show()
