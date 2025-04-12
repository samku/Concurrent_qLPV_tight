from imports import *
from utils.qLPV_BFR import qLPV_BFR

def sequential_benchmark(dataset,models,id_params):
    
    #Extract from modelset
    model_LPV = models['model_LPV']
    RCI_LPV = models['RCI_LPV']
    sizes = models['sizes']
    kappa = models['kappa']
    N_MPC = models['N_MPC']
    activation_func = models['activation']

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
    Zs_observer = np.hstack((Us_observer, Ys_observer))

    # Optimization params
    eta = id_params['eta']
    rho_th = id_params['rho_th']
    adam_epochs = id_params['adam_epochs']
    lbfgs_epochs = id_params['lbfgs_epochs']
    kappa_x = id_params['kappa_x']

    #Constraints
    F = RCI_LPV['F']
    E = RCI_LPV['E']
    V = RCI_LPV['V']
    m = F.shape[0]
    m_bar = len(V)
    HY = constraints['HY']
    hY = constraints['hY']
    mY = HY.shape[0]
    Y_set = Polytope(A=HY,b=hY)
    Y_vert = Y_set.V
    vY = len(Y_vert)
    HU = constraints['HU']
    hU = constraints['hU']
    mU = HU.shape[0]


    #Constraints
    #HY = constraints['HY']
    #hY = constraints['hY']
    #mY = HY.shape[0]
    #Y_set = Polytope(A=HY,b=hY)
    #Y_vert = Y_set.V
    #vY = len(Y_vert)
    #HU = constraints['HU']
    #hU = constraints['hU']
    #mU = HU.shape[0]

    # Store previous parameters
    A_prev = model_LPV['A'].copy()
    B_prev = model_LPV['B'].copy()
    C_prev = model_LPV['C'].copy()
    L_prev = model_LPV['L'].copy()
    Win_prev = model_LPV['Win'].copy()
    bin_prev = model_LPV['bin'].copy()
    Whid_prev = model_LPV['Whid'].copy()
    bhid_prev = model_LPV['bhid'].copy()
    Wout_prev = model_LPV['Wout'].copy()
    bout_prev = model_LPV['bout'].copy()

    jax.config.update("jax_enable_x64", True)

    # Define the optimization variables
    A = jnp.array(model_LPV['A'].copy())
    B = jnp.array(model_LPV['B'].copy())
    C = jnp.array(model_LPV['C'].copy())
    Win = jnp.array(model_LPV['Win'].copy())
    bin = jnp.array(model_LPV['bin'].copy())
    Whid = jnp.array(model_LPV['Whid'].copy())
    bhid = jnp.array(model_LPV['bhid'].copy())
    Wout = jnp.array(model_LPV['Wout'].copy())
    bout = jnp.array(model_LPV['bout'].copy())
    
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
    def SS_forward(x, u, params):
        y_next = output_fcn(x, u, params)
        x_next = state_fcn(x, u, params)
        return x_next, jnp.hstack((x, y_next))
    
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
    model.loss(rho_th=rho_th, train_x0 = True, custom_regularization=custom_regularization)  
    model.optimization(adam_eta=eta, adam_epochs=adam_epochs, lbfgs_epochs=lbfgs_epochs, memory=100, iprint=10)
    model.fit(Ys_train, Us_train)
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

    #Get prediction error
    BFR_train_qLPV_CL, y_train_qLPV_CL, x0_train_qLPV_CL = qLPV_BFR(model_LPV, Us_train, Ys_train, observer = False, activation_func = activation_func)
    BFR_test_qLPV_CL, y_test_qLPV_CL, x0_test_qLPV_CL = qLPV_BFR(model_LPV, [Us_test], [Ys_test], observer = False, activation_func = activation_func)
    BFR_observer_qLPV_CL, y_observer_qLPV_CL, x0_observer_qLPV_CL = qLPV_BFR(model_LPV, [Us_observer], [Ys_observer], observer = False, activation_func = activation_func)
    print('BFR train: qLPV CL', BFR_train_qLPV_CL, 'qLPV CL', BFR_test_qLPV_CL, 'qLPV CL', BFR_observer_qLPV_CL)


    #Extract disturbance set
    params_updated = [A_new, B_new, C_new, Win_new, bin_new, Whid_new, bhid_new, Wout_new, bout_new]
    from predict_initial_states import predict_initial_states   
    x0_guess = predict_initial_states(model_LPV, [Us_observer], [Ys_observer], activation=activation_func)
    simulator = partial(SS_forward, params=params_updated)
    xy_next = jax.lax.scan(simulator, jnp.array(x0_guess[0]), Us_observer)[1]
    x_observer = xy_next[:,0:nx]
    y_next = xy_next[:,nx:]
    error = Ys_observer-y_next
    w_hat = (jnp.max(error,axis=0)+jnp.min(error,axis=0))/2
    eps_w = (jnp.max(error,axis=0)-jnp.min(error,axis=0))/2
    w_hat = np.array(w_hat)
    eps_w = kappa*np.array(eps_w)
    print('eps_w', eps_w)
    print('w_hat', w_hat)

    #Compute maximal RCI set
    hY_tight = hY-(HY@w_hat+np.abs(HY)@eps_w)
    backreach = []
    backreach.append(Polytope(A=HY@C_new,b=hY_tight))
    num_iters = 18
    volume_set = []
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    for t in range(num_iters):
        H_prev = backreach[t].A
        h_prev = backreach[t].b

        H_intersected = np.vstack((H_prev, np.eye(nx), -np.eye(nx)))
        h_intersected = np.hstack((h_prev, np.array([100.,100.,100.,100.])))
        intersected_set = Polytope(A=H_intersected,b=h_intersected)
        

        H_xu = np.hstack((HY@C_new, np.zeros((mY,nu))))
        h_xu = hY_tight
        H_xu = np.vstack((H_xu, np.hstack((np.zeros((mU,nx)), HU))))
        h_xu = np.hstack((h_xu, hU))
        for i in range(nq):
            H_xu = np.vstack((H_xu, np.hstack((H_prev@A_new[i], H_prev@B_new[i]))))
            h_xu = np.hstack((h_xu, h_prev))
        full_set = Polytope(A=H_xu,b=h_xu)
        projected_set = full_set.projection(project_away_dim = np.arange(nx,nx+nu))
        backreach.append(projected_set)
        if t<nx:
            volume_set.append(np.inf) #Unbounded set
        else:
            backreach[-1].plot(ax=ax1, patch_args={'alpha':0.2})
            volume_set.append(projected_set.volume())
            ax2.scatter(t, volume_set[t], color='red')
        plt.pause(0.1)


    #Evaluate cost of RCI set
    H_RCI = backreach[-1].A
    h_RCI = backreach[-1].b
    opti = ca.Opti()
    V_mean = np.mean(V, axis = 0)
    U_mats = []
    for i in range(m_bar):
        U_loc = np.zeros((nu,nu*m_bar))
        U_loc[:,i*nu:(i+1)*nu] = np.eye(nu)
        U_mats.append(U_loc)
    U_mats = np.array(U_mats)

    A_mean = np.mean(A_new, axis=0)
    B_mean = np.mean(B_new, axis=0)
    opti = ca.Opti()
    cost_traj = 0.
    x_traj = opti.variable(nx*mY, N_MPC+1)
    u_traj = opti.variable(nu*mY, N_MPC)
    for i in range(mY):
        x_traj_loc = x_traj[nx*i:nx*(i+1),:]
        u_traj_loc = u_traj[nu*i:nu*(i+1),:]
        opti.subject_to(x_traj_loc[:,0]==np.zeros((nx)))
        for t in range(N_MPC):
            opti.subject_to(x_traj_loc[:,t+1]==A_mean@x_traj_loc[:,t]+B_mean@u_traj_loc[:,t])
            opti.subject_to(H_RCI@x_traj_loc[:,t]<=h_RCI)
            opti.subject_to(HU@u_traj_loc[:,t]<=hU)
            vector = C_new@x_traj_loc[:,t]-Y_vert[i]
            cost_traj = cost_traj+ca.dot(vector,vector)
        opti.subject_to(H_RCI@x_traj_loc[:,N_MPC]<=h_RCI)
        vector = C_new@x_traj_loc[:,N_MPC]-Y_vert[i]
        cost_traj = cost_traj+ca.dot(vector,vector)
    opti.minimize(cost_traj)
    opti.solver('ipopt')
    sol = opti.solve()
    x_traj = sol.value(x_traj)
    u_traj = sol.value(u_traj)
    cost = sol.value(cost_traj)
    print('Cost of RCI set', cost)

    for i in range(mY):
        x_piece = x_traj[nx*i:nx*(i+1),:]
        ax1.plot(x_piece[0,:],x_piece[1,:], color='red')
        y_plot = C_new[0]@x_piece
        print(y_plot)
        ax3.plot(y_plot, color='red')
    plt.show()

    F = H_RCI.copy()
    RCI_concur = RCI_LPV.copy()
    RCI_concur['F'] = H_RCI
    RCI_concur['yRCI'] = h_RCI
    Xt_vert = Polytope(A=H_RCI,b=h_RCI).V
    m = H_RCI.shape[0]
    m_bar = len(Xt_vert)

    #Vertex maps
    V = [] 
    all_combinations = list(combinations(range(m), nx))
    for i in range(m_bar): #Keep index of input and state the same
        for j in range(len(all_combinations)):
            id_loc = np.array(all_combinations[j])
            V_test = H_RCI[id_loc]
            h_test = h_RCI[id_loc]
            if np.linalg.norm(V_test @ Xt_vert[i] - h_test , np.inf)<=1e-5:
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
    RCI_concur['E'] = E
    RCI_concur['V'] = V
    RCI_concur['W'] = np.vstack((w_hat, eps_w))

    #Compute tracking bounds
    

    return model_LPV, RCI_concur




    



