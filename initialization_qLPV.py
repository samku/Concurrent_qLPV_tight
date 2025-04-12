from imports import *
from utils.qLPV_BFR import qLPV_BFR

def initialization(dataset, sizes, kappa, id_params, activation_func = 1):
    
    if activation_func == 1:
        activation = nn.swish
    elif activation_func == 2:
        activation = nn.relu
    elif activation_func == 3:
        @jax.jit
        def activation(x):
            return nn.elu(x)+1.
    elif activation_func == 4:
        activation = nn.sigmoid


    #Extract data
    Y_train = dataset['Y_train']
    Y_test = dataset['Y_test']
    Ys_train = dataset['Ys_train']
    Us_train = dataset['Us_train']
    Ys_observer = dataset['Ys_observer']
    Us_observer = dataset['Us_observer']
    Ys_test = dataset['Ys_test']
    Us_test = dataset['Us_test']
    ny = Ys_train[0].shape[1]
    nu = Us_train[0].shape[1]
    nx = sizes[0]
    nq = sizes[1]
    nth = sizes[2]
    nH = sizes[3]
    constraints = dataset['constraints']    

    # Optimization params
    iprint = id_params['iprint']
    memory = id_params['memory']
    eta = id_params['eta']
    rho_th = id_params['rho_th']
    adam_epochs = id_params['adam_epochs']
    lbfgs_epochs = id_params['lbfgs_epochs']
    train_x0 = id_params['train_x0']
    weight_RCI = id_params['weight_RCI']
    N_MPC = id_params['N_MPC']
    kappa_x = id_params['kappa_x']

    jax.config.update("jax_enable_x64", True)

    #Extract sizes
    print('Identifying model with nx:', nx, 'nq:', nq, 'nth:', nth, 'nH:', nH)
    model = LinearModel(nx, ny, nu, feedthrough=False)
    model.loss(rho_x0=0.000, rho_th=0.0001, train_x0=True)
    model.optimization(adam_eta=0.001, adam_epochs=2000, lbfgs_epochs=2000)
    model.fit(Ys_train, Us_train)
    A_LTI, B_LTI, C_LTI, D_LTI = model.ssdata()
    A_LTI = np.array(A_LTI.copy())
    B_LTI = np.array(B_LTI.copy())
    C_LTI = np.array(C_LTI.copy())


    # Define the optimization variables
    key = jax.random.PRNGKey(10)
    key1, key2, key3 = jax.random.split(key, num=3)
    A = 0.0001*jax.random.normal(key1, (nq,nx,nx))
    B = 0.0001*jax.random.normal(key2, (nq,nx,nu))
    C = 0.0001*jax.random.normal(key2, (ny,nx))
    Win = 0.0001*jax.random.normal(key1, (nq, nth, nx))
    bin = 0.0001*jax.random.normal(key2, (nq, nth))
    Whid = 0.0001*jax.random.normal(key3, (nq, nH-1, nth, nth))
    bhid = 0.0001*jax.random.normal(key1, (nq, nH-1, nth))
    Wout = 0.0001*jax.random.normal(key2, (nq, nth))
    bout = 0.0001*jax.random.normal(key3, (nq, ))

    for i in range(nq):
        A = A.at[i].set(jnp.array(A_LTI))
        B = B.at[i].set(jnp.array(B_LTI))
    C = jnp.array(C_LTI)

    @jax.jit
    def parameter_fcn(x,u,Win,bin,Whid,bhid,Wout,bout):
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
    def state_fcn(x,u,params):
        A, B, C, Win, bin, Whid, bhid, Wout, bout = params
        p = parameter_fcn(x,u,Win,bin,Whid,bhid,Wout,bout)
        A_tot = jnp.zeros((nx,nx))  
        B_tot = jnp.zeros((nx,nu))
        for i in range(nq):
            A_tot += A[i]*p[i]
            B_tot += B[i]*p[i]
        return A_tot@x+B_tot@u

    @jax.jit
    def output_fcn(x,u,params):
        A, B, C, Win, bin, Whid, bhid, Wout, bout = params
        return C@x

    @jax.jit
    def SS_forward_output(x, u, params):
        A, B, C, Win, bin, Whid, bhid, Wout, bout = params
        y_current = output_fcn(x, u, params)
        x_next = state_fcn(x, u, params).reshape(-1)
        return x_next, y_current
    
    @jax.jit
    def custom_regularization(params,x0): 
        custom_R = 0.
        A, B, C, Win, bin, Whid, bhid, Wout, bout = params
        A_mean = jnp.mean(A, axis=0)
        B_mean = jnp.mean(B, axis=0)
        for i in range(nq):
            A_deviation = A[i]-A_mean
            B_deviation = B[i]-B_mean
            custom_R = custom_R+kappa_x*jnp.sum(A_deviation**2)
            custom_R = custom_R+kappa_x*jnp.sum(B_deviation**2)
        return custom_R
    
    #Pass to optimizer
    model = Model(nx, ny, nu, state_fcn=state_fcn, output_fcn=output_fcn, Ts=1)
    model.init(params=[A, B, C, Win, bin, Whid, bhid, Wout, bout])
    model.loss(rho_th=rho_th, train_x0 = train_x0, custom_regularization=custom_regularization)  
    model.optimization(adam_eta=eta, adam_epochs=adam_epochs, lbfgs_epochs=lbfgs_epochs, memory=memory, iprint=iprint)
    model.fit(Ys_train, Us_train)
    identified_params = model.params
    A_new = np.array(model.params[0])
    B_new = np.array(model.params[1])
    C_new = np.array(model.params[2])
    Win_new = np.array(model.params[3])
    bin_new = np.array(model.params[4])
    Whid_new = np.array(model.params[5])
    bhid_new = np.array(model.params[6])
    Wout_new = np.array(model.params[7])
    bout_new = np.array(model.params[8])
    model_LPV = {'A': A_new, 'B': B_new, 'C': C_new, 'Win': Win_new, 'bin': bin_new, 'Whid': Whid_new, 'bhid': bhid_new, 'Wout': Wout_new, 'bout': bout_new}
    model_LPV['L'] = np.zeros((nq, nx, ny))

    print(A_new)
    print(B_new)
    BFR, _, _ = qLPV_BFR(model_LPV, Us_train, Ys_train, observer = False, activation_func = activation_func)
    print('Init BFR:', BFR)

    params_sys = [A_new, B_new, C_new, Win_new, bin_new, Whid_new, bhid_new, Wout_new, bout_new]
    simulator = partial(SS_forward_output, params=params_sys)
    Y_hat_observer = jax.lax.scan(simulator, jnp.zeros((nx,)), Us_observer)[1]
    error_observer = Ys_observer-Y_hat_observer
    max_error_observer = np.max(error_observer,axis=0)
    min_error_observer = np.min(error_observer,axis=0)
    w0 = 0.5*(max_error_observer+min_error_observer)
    epsw = 0.5*(max_error_observer-min_error_observer)

    #Build RCI set
    HY = constraints['HY']
    hY = constraints['hY']
    HU = constraints['HU']
    hU = constraints['hU']
    hY_tight = hY - (HY @ w0 + kappa*np.abs(HY) @ epsw)
    Y_set = Polytope(A = HY, b = hY)
    Y_vert = Y_set.V
    mY = len(Y_vert)

    
    A_mean = np.mean(A_new, axis=0)
    B_mean = np.mean(B_new, axis=0)
 
    #Hardcoded $\tilde{F}$ matrix
    #For 2D, can select number of facets of template
    #Similar to A One-step Approach to Computing a Polytopic Robust Positively Invariant Set, P.Trodden
    m = 4
    if nx == 2:
        angle = 2*np.pi/m
        F = np.array([]).reshape(0, nx)
        for i in range(m):
            row_add = [[np.cos(i*angle), np.sin((i)*angle)]]
            F = np.concatenate((F, row_add), axis=0)
    else:
        F = np.vstack((np.eye(nx), -np.eye(nx)))
    
    m = F.shape[0]
    y0 = np.ones(m)

    if 0:
        F = np.vstack((-np.eye(nx), np.ones((1,nx))))
        y0 = np.hstack((np.zeros((nx,)), 1))-F@(0.5*np.ones((nx,)))/nx
        m = F.shape[0]
        y0 = np.ones(m)

    X_template = Polytope(A = F, b = y0)
    Xt_vert = X_template.V
    m_bar = len(Xt_vert)
    print('Identifying RCI set with ', m, ' faces and ', m_bar, ' vertices')
    
    opti = ca.Opti()
    WM_init = opti.variable(nx,nx)
    WMinv_init = ca.inv(WM_init)
    opti.set_initial(WM_init, np.eye(nx))
    uRCI_init = opti.variable(nu, m_bar)
    u_bound_mod = opti.variable(nu)
    du_bound = hU[:nu] - u_bound_mod
    cost_bound = ca.dot(du_bound, du_bound)
    hU_modified = ca.vertcat(u_bound_mod, u_bound_mod) #Enable relaxation slightly to improve chances of feasibility

    for k in range(m_bar):
        for i in range(nq):
            vector = F @ (WMinv_init @ (A_new[i] @ WM_init @ Xt_vert[k] + B_new[i] @ uRCI_init[:,k])) - y0
            opti.subject_to(vector <= 0)
        vector = HY @ C_new @ WM_init @ Xt_vert[k] - hY_tight
        opti.subject_to(vector <= 0)
        vector = HU @ uRCI_init[:,k] - hU
        opti.subject_to(vector <= 0)

    #Size
    cost = 0.
    x_traj = opti.variable(nx*mY, N_MPC+1)
    u_traj = opti.variable(nu*mY, N_MPC)
    for i in range(mY):
        x_traj_loc = x_traj[nx*i:nx*(i+1),:]
        u_traj_loc = u_traj[nu*i:nu*(i+1),:]
        opti.subject_to(x_traj_loc[:,0]==np.zeros((nx)))
        for t in range(N_MPC):
            opti.subject_to(x_traj_loc[:,t+1]==A_mean@x_traj_loc[:,t]+B_mean@u_traj_loc[:,t])
            opti.subject_to(F@WMinv_init@x_traj_loc[:,t]<=y0)
            opti.subject_to(HU@u_traj_loc[:,t]<=hU)
            vector = C_new@x_traj_loc[:,t]-Y_vert[i]
            cost = cost+ca.dot(vector,vector)
        opti.subject_to(F@WMinv_init@x_traj_loc[:,N_MPC]<=y0)
        vector = C_new@x_traj_loc[:,N_MPC]-Y_vert[i]
        cost = cost+ca.dot(vector,vector)
    

    opti.minimize(cost+1000*cost_bound)
    opti.solver('ipopt')
    sol = opti.solve()
    WM = sol.value(WM_init)
    WMinv = sol.value(WMinv_init)
    uRCI = sol.value(uRCI_init)
    if nu==1:
        uRCI = np.reshape(uRCI, (1, m_bar))
    hU_modified = sol.value(hU_modified)
    cost = sol.value(cost)
    print(hU_modified)
    print(hU)
    
    #Construct vertex maps and configuration constraints
    F = F @ WMinv
    #Vertex maps
    V = [] 
    all_combinations = list(combinations(range(m), nx))
    for i in range(m_bar): #Keep index of input and state the same
        for j in range(len(all_combinations)):
            id_loc = np.array(all_combinations[j])
            V_test = F[id_loc]
            h_test = y0[id_loc]
            if np.linalg.norm(V_test @ WM @ Xt_vert[i] - h_test , np.inf)<=1e-5:
                ones_mat = np.zeros((nx,m))
                for k in range(nx):
                    ones_mat[k, id_loc[k]] = 1
                V.append(np.linalg.inv(V_test) @ ones_mat)

    E = np.array([]).reshape(0, m) #Configuration constraints
    for k in range(m_bar):
        local_mat = F @ V[k] - np.eye(m)
        for j in range(m):
            E = np.concatenate((E, [local_mat[j]]), axis = 0)
    cc_cone = Polytope(A = E, b = np.zeros((E.shape[0],)))
    E_reduced = cc_cone.A
    E = E_reduced.copy()

    RCI_dict = {}
    RCI_dict['F'] = F   
    RCI_dict['V'] = V
    RCI_dict['E'] = E
    RCI_dict['yRCI'] = y0
    RCI_dict['uRCI'] = uRCI
    RCI_dict['cost'] = sol.value(cost)
    RCI_dict['W'] = np.vstack((w0, epsw))

    return model_LPV, RCI_dict