from imports import *
from utils.qLPV_BFR import qLPV_BFR

def concurrent_identification(dataset,models,id_params,use_bounds = True, zeta = 1.05, tightened_W = False, k_max = 5):
    
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
    rho_a = id_params['rho_a']
    adam_epochs = id_params['adam_epochs']
    weight_RCI = id_params['weight_RCI']
    regularization_QP = id_params['regularization_QP']
    
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
    L = jnp.array(model_LPV['L'].copy())    
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
        A, B, L, C, Win, bin, Whid, bhid, Wout, bout = params
        p = parameter_fcn(x,u,Win,bin,Whid,bhid,Wout,bout)
        A_tot = jnp.zeros((nx,nx))  
        B_tot = jnp.zeros((nx,nu))
        for i in range(nq):
            A_tot += A[i]*p[i]
            B_tot += B[i]*p[i]
        return A_tot@x+B_tot@u
    
    @jax.jit
    def state_fcn_obsv(x,z,params):
        A, B, L, C, Win, bin, Whid, bhid, Wout, bout = params
        p = parameter_fcn(x,z[0:nu],Win,bin,Whid,bhid,Wout,bout)
        A_tot = jnp.zeros((nx,nx))  
        B_tot = jnp.zeros((nx,nu))
        L_tot = jnp.zeros((nx,ny))
        for i in range(nq):
            A_tot += A[i]*p[i]
            B_tot += B[i]*p[i]
            L_tot += L[i]*p[i]
        return (A_tot-L_tot@C)@x+jnp.hstack((B_tot,L_tot))@z
    
    @jax.jit
    def output_fcn(x,u,params):
        A, B, L, C, Win, bin, Whid, bhid, Wout, bout = params
        return C@x
    
    @jax.jit
    def SS_forward(x, u, params):
        y_next = output_fcn(x, u, params)
        x_next = state_fcn(x, u, params)
        return x_next, jnp.hstack((x, y_next))
    
    @jax.jit
    def SS_forward_output(x, z, params):
        #This is on the observer
        y_next = output_fcn(x, z, params)
        #x_next = state_fcn_obsv(x, z, params)
        x_next = state_fcn(x, z[:nu], params)
        return x_next, jnp.hstack((x, y_next))
    
    @jax.jit
    def get_parameter_samples(carry,multiplier,vertices, params):
        A, B, L, C, Win, bin, Whid, bhid, Wout, bout = params
        xu = vertices@multiplier
        x = xu[0:nx]
        u = xu[nx:nx+nu]
        p = parameter_fcn(x,u,Win,bin,Whid,bhid,Wout,bout)
        return carry, p
    
    @jax.jit
    def bound_propagation(params, yRCI_comp):
        A, B, L, C, Win, bin, Whid, bhid, Wout, bout = params
        RCI_vertices = jnp.zeros((0,nx))
        for i in range(m_bar):
            RCI_vertices = jnp.vstack((RCI_vertices, V[i]@yRCI_comp))

        max_RCI_vertices = jnp.max(RCI_vertices,axis=0)
        min_RCI_vertices = jnp.min(RCI_vertices,axis=0)
        mean_input = 0.5*(max_RCI_vertices+min_RCI_vertices)
        var_input = 0.5*(max_RCI_vertices-min_RCI_vertices)+zeta
                
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
    def RCI_computation_single(params, yRCI_prev, factor=1):
        A, B, L, C, Win, bin, Whid, bhid, Wout, bout = params 

        #Extract bounds
        if use_bounds == True:
            a_used, mean_input, var_input = bound_propagation(params, yRCI_prev)
        else:
            a_used = jnp.zeros((nq,))
            mean_input = jnp.zeros((nx,))
            var_input = 100*jnp.ones((nx,))
        a_used = a_used*factor
        bound_lb = mean_input-var_input
        bound_ub = mean_input+var_input

        #Extract disturbance set parameters
        simulator = partial(SS_forward_output, params=params)
        xy_next = jax.lax.scan(simulator, jnp.zeros((nx,)), Zs_observer)[1]
        
        x_observer = xy_next[:,0:nx]
        y_next = xy_next[:,nx:]
        if tightened_W == True:
            par_get_x_in_RCI = partial(get_x_in_B, bound_lb = bound_lb, bound_ub = bound_ub)
            x_ids = jax.lax.scan(par_get_x_in_RCI, jnp.zeros((0,)), x_observer)[1]            
            yhat_in = y_next*x_ids
            y_in = Ys_observer*x_ids
            error = y_in-yhat_in
            w_hat = (jnp.max(error,axis=0)+jnp.min(error,axis=0))/2
            eps_w = (jnp.max(error,axis=0)-jnp.min(error,axis=0))/2
        else:
            error = Ys_observer-y_next
            w_hat = (jnp.max(error,axis=0)+jnp.min(error,axis=0))/2
            eps_w = (jnp.max(error,axis=0)-jnp.min(error,axis=0))/2

        h_incr = jnp.hstack((var_input,var_input))+H_incr@mean_input
        
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
        if use_bounds == True:
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
        else:
            A_ineq = jnp.vstack((A_ineq,
                                 jnp.hstack((FA_part, FB_part, Fb_part)),
                                jnp.hstack((HC_part, HD_part, Hb_part)),
                                jnp.hstack((U1_part, U2_part, Ub_part)),
                                jnp.hstack((CC_1, CC_2, CC_3))))
            b_ineq = jnp.hstack((b_ineq, part_1, part_2, part_3,part_5))

        num_vars = m+nu*m_bar+vY*N_MPC*nu
        Q = Q1+regularization_QP*np.eye(num_vars)
        c = c1
        A_eq = jnp.zeros((0,num_vars))
        b_eq = jnp.zeros((0,))
 
        QP_soln = qpax.solve_qp_primal(Q, c, A_eq, b_eq, A_ineq, b_ineq, solver_tol=1e-8, target_kappa=1e-8)
        
        yRCI_comp = QP_soln[0:m]
        uRCI_comp = QP_soln[m:m+nu*m_bar].reshape(m_bar,nu)
        y_trajec = Z_total@QP_soln
        error = target_y_vertices - y_trajec
        cost = jnp.dot(error,error)

        a_new, _, _ = bound_propagation(params, yRCI_comp)

        return cost, yRCI_comp, uRCI_comp, w_hat, eps_w, a_new, a_used
    
    @jax.jit
    def RCI_computation_array(y_prev, index, params):
        cost, y_next, u_next, w_hat, eps_w, a_new, a_used = RCI_computation_single(params, y_prev)
        return y_next, [cost, jnp.reshape(u_next, (1, -1))[0], w_hat, eps_w, a_new, a_used]
      
    @jax.jit
    def RCI_computation(params, yRCI_prev):
        if k_max>0:
            par_RCI_computation_array = partial(RCI_computation_array, params=params)
            y_final, cost_W = jax.lax.scan(par_RCI_computation_array, yRCI_prev, jnp.arange(k_max))
            costs = cost_W[0][-1]
            yRCI_final = y_final
            uRCI_final = jnp.reshape(cost_W[1][-1], (m_bar,nu))
            w_hats = cost_W[2][-1]
            eps_ws = cost_W[3][-1]
            a_news = cost_W[4][-1]
            a_useds = cost_W[5][-1]
        else:
            costs, yRCI_final, uRCI_final, w_hats, eps_ws, a_news, a_useds = RCI_computation_single(params, yRCI_prev)
        return costs, yRCI_final, uRCI_final, w_hats, eps_ws, a_news, a_useds
    
    params_sys = [A, B, L, C, Win, bin, Whid, bhid, Wout, bout]
    cost_prev, yRCI_prev, uRCI_prev, _, _, a_new, a_used  = RCI_computation_single(params_sys, 100*jnp.ones((m,)), factor=1)
    yRCI_prev = np.array(yRCI_prev.copy())
    uRCI_prev = np.array(uRCI_prev.copy())
    yRCI_pb_o = yRCI_prev.copy()
    uRCI_pb_o = uRCI_prev.copy()     

    _, _, _, _, _, _, _ = RCI_computation(params_sys, yRCI_prev)

    @jax.jit
    def gradient_RCI_computation(params, yRCI_prev):
        cost_RCI, yRCI, uRCI, _, _, a_new, _ = RCI_computation(params, yRCI_prev)
        return cost_RCI
    
    RCI_gradient = jit(grad(gradient_RCI_computation, argnums=0))

    def clip_gradients(gradients, threshold):
        total_norm = np.sqrt(sum(np.linalg.norm(g)**2 for g in gradients))
        if total_norm > threshold:
            scale = threshold / total_norm
        else:
            scale = 1.
        return total_norm, scale
          
    @jax.jit
    def custom_regularization(params,yRCI_prev):  
        cost_RCI, yRCI, uRCI, _, _, a_new, _ = RCI_computation(params, yRCI_prev)
        l2_loss = jnp.sum((params[2])**2)
        return cost_RCI, l2_loss, a_new
    
    def single_loss(U,Y,x0,params):
        simulator_PE = partial(SS_forward, params=params)
        xy_hat = jax.lax.scan(simulator_PE, x0, U)[1]
        y_hat = xy_hat[:,nx:]
        loss = jnp.sum((Y-y_hat)**2)/len(Y)
        return loss
    
    def mini_line_search(parameters_current, update_direction, current_loss, eta, sysID, yRCI_prev, alpha=0.1, tau=0.5, min_eta=1e-6):
        total_norm_sq = sum(np.linalg.norm(u)**2 for u in update_direction)        
        step_size = eta
        while step_size >= min_eta:
            candidate_parameters = [p - step_size * u for p, u in zip(parameters_current, update_direction)]
            new_loss = sysID(candidate_parameters, yRCI_prev)
            if new_loss <= current_loss - alpha * step_size * total_norm_sq:
                return step_size
            step_size *= tau 
        return step_size
    
    vmap_single_loss = jax.vmap(single_loss, in_axes=(0,0,0,None))
    vmap_single_loss = jax.jit(vmap_single_loss)
    Us_train_jax = jnp.array(Us_train.copy())
    Ys_train_jax = jnp.array(Ys_train.copy())
    num_datasets = len(Ys_train)
    @jax.jit
    def prediction_loss(params, x0):
        return jnp.sum(vmap_single_loss(Us_train_jax, Ys_train_jax, x0, params))/num_datasets

    #Define identification function
    @jax.jit
    def sysID(params_x0, yRCI_prev):

        [A,B,L,C,Win,bin,Whid,bhid,Wout,bout,x0] = params_x0
        params_model = [A,B,L,C,Win,bin,Whid,bhid,Wout,bout]

        #Prediction error
        loss_prediction = prediction_loss(params_model, x0)
   
        #Regularization
        cost_RCI, l2_loss, a_new = custom_regularization(params_model, yRCI_prev)

        #Total loss
        total_loss = loss_prediction + weight_RCI*cost_RCI + rho_th*l2_loss - rho_a*jnp.sum(a_new)

        return total_loss

    SysID_gradient = jit(grad(sysID, argnums=0))
    from predict_initial_states import predict_initial_states   
    model_qLPV = {'A': A, 'B': B, 'C': C, 'L': L, 'Win': Win, 'bin': bin, 'Whid': Whid, 'bhid': bhid, 'Wout': Wout, 'bout': bout}
    x0_guess = predict_initial_states(model_qLPV, Us_train, Ys_train, activation=activation_func)
    parameters_init = [A,B,L,C,Win,bin,Whid,bhid,Wout,bout,jnp.array(x0_guess)]

    params_sys = [A,B,L,C,Win,bin,Whid,bhid,Wout,bout]
    grads_init = SysID_gradient(parameters_init, yRCI_prev)


    #Do adam for some steps
    num_vars = len(parameters_init)
    vv = [np.zeros(zi.shape) for zi in parameters_init]
    mm = [np.zeros(zi.shape) for zi in parameters_init]
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    print_interval = 10
    cost_history = []  
    best_cost_history = []
    best_parameters = parameters_init.copy()
    best_index = []
    yRCI_pb = yRCI_prev.copy()
    parameters_current = parameters_init.copy()
    print('zeta:', zeta)
    print('k_max:', k_max)
    print('Adam steps:')
    iters_tqdm = tqdm(total=adam_epochs, desc='Iterations', ncols=30, bar_format='{percentage:3.0f}%|{bar}|', leave=True, position=0)
    loss_log = tqdm(total=0, position=1, bar_format='{desc}')
    for i in range(adam_epochs):
        sysID_loss = sysID(parameters_current, yRCI_prev)
        cost_history.append(sysID_loss)

        #Keep track of best cost and parameters
        if i==0:
            best_cost_history.append(sysID_loss)
            best_index.append(i)
            best_parameters = parameters_current.copy()
            #Update best parameters
            params_model_best = best_parameters[:-1]
            x0_model_best = best_parameters[-1]
            cost_RCI_best, _, _, w_hat_best, epsw_best, _, _ = RCI_computation(params_model_best, yRCI_pb)
            loss_prediction_best = prediction_loss(params_model_best, x0_model_best)
        else:
            if sysID_loss < best_cost_history[-1]:
                best_cost_history.append(sysID_loss)
                best_index.append(i)
                best_parameters = parameters_current.copy()
                yRCI_pb = yRCI_prev.copy()
                uRCI_pb = uRCI_prev.copy()
                #Update best parameters
                params_model_best = best_parameters[:-1]
                x0_model_best = best_parameters[-1]
                cost_RCI_best, _, _, w_hat_best, epsw_best, _, _ = RCI_computation(params_model_best, yRCI_pb)
                loss_prediction_best = prediction_loss(params_model_best, x0_model_best)

            else:
                best_cost_history.append(best_cost_history[-1])
                best_index.append(best_index[-1])

        #RCI gradient
        gradient_check = RCI_gradient(parameters_current[:-1], yRCI_prev)
        gradient_norm, gradient_scale = clip_gradients(gradient_check, 500.)

        #Do gradient descent update
        gradients = SysID_gradient(parameters_current, yRCI_prev)
        #gradients = [gg * gradient_scale for gg in gradients]
        update_direction = []
        for j in range(num_vars):
            mm[j] = beta1 * mm[j] + (1 - beta1) * gradients[j]
            vv[j] = beta2 * vv[j] + (1 - beta2) * (gradients[j] ** 2)
            m_hat = mm[j] / (1 - beta1)
            v_hat = vv[j] / (1 - beta2)
            update_direction.append(m_hat / (jnp.sqrt(v_hat) + epsilon))
            #parameters_current[j] = parameters_current[j] - eta * m_hat / (jnp.sqrt(v_hat) + epsilon)

        if i>15000:
            #adaptive_eta = mini_line_search(parameters_current, update_direction, sysID_loss, eta, sysID, yRCI_prev)
            adaptive_eta = 0.0001
            if gradient_scale<1.:
                adaptive_eta = 0.00001
        else:
            adaptive_eta = eta
        for j in range(num_vars):
            parameters_current[j] = parameters_current[j] - adaptive_eta * update_direction[j]

        parameters_current_model = parameters_current[:-1]

        #Update yRCI and uRCI previous for BP
        cost_RCI_curr, yRCI_current, uRCI_current, w_hat_best, epsw_best, a_new, a_used = RCI_computation(parameters_current_model, yRCI_prev) 
        yRCI_prev = np.float64(yRCI_current.copy())
        uRCI_prev = np.float64(uRCI_current.copy())
        a_rounded = np.round(a_new, 3)

        if jnp.isnan(cost_RCI_curr):
            print("Stopping computation: cost_RCI_curr is NaN at iteration", i)
            break

        str = f"    C = {sysID_loss: 10.6f}, B = {best_cost_history[-1]: 8.6f}, Rb = {cost_RCI_best: 2.3f}, a = {a_rounded}, Dg = {gradient_norm: 2.3f}, ||| RCIc = {weight_RCI*cost_RCI_curr: 8.6f}, loss = {loss_prediction_best: 8.6f}"
        str += f", Iter = {i+1} of {adam_epochs}"
        loss_log.set_description_str(str)
        iters_tqdm.update(1)

    loss_log.close()
    iters_tqdm.close()

    parameters_optimized = best_parameters.copy()
    
    [A_new, B_new, L_new, C_new, Win_new, bin_new, Whid_new, bhid_new, Wout_new, bout_new, x0_new] = parameters_optimized
    
    model_LPV_concur = {'A': np.float64(A_new),
                        'B': np.float64(B_new),
                        'L': np.float64(L_new),
                        'C': np.float64(C_new),
                        'Win': np.float64(Win_new),
                        'bin': np.float64(bin_new),
                        'Whid': np.float64(Whid_new),
                        'bhid': np.float64(bhid_new),
                        'Wout': np.float64(Wout_new),
                        'bout': np.float64(bout_new)}
    

    #Reevalute RCI
    RCI_concur = RCI_LPV.copy()
    params_model_old = [A,B,L,C,Win,bin,Whid,bhid,Wout,bout]
    params_model_new = [A_new, B_new, L_new, C_new, Win_new, bin_new, Whid_new, bhid_new, Wout_new, bout_new]


    costRCI_old, yRCI_old, uRCI_old, w_hat_old, epsw_old, a_new_old, a_used_old = RCI_computation(params_model_old, yRCI_pb_o)
    costRCI_new, yRCI_new, uRCI_new, w_hat_new, epsw_new, a_new_new, a_used_new = RCI_computation(params_model_new, yRCI_pb)
    RCI_concur['yRCI'] = np.array(yRCI_new.copy()) 
    RCI_concur['uRCI'] = np.array(uRCI_new.copy())
    RCI_concur['bRCI'] = 0.
    RCI_concur['cost'] = costRCI_new    
    RCI_concur['W'] = np.vstack((w_hat_new, epsw_new))
    RCI_concur['a'] = a_new_old
    print('Original cost:', costRCI_old)
    print('Updated cost:', costRCI_new)
    print('a_new:', a_new_new)
    print('y_new:', yRCI_new)
    print('u_new:', uRCI_new)

    #Verify parameter samples
    RCI_vertices = np.zeros((0,nx))
    for i in range(m_bar):
        RCI_vertices = np.vstack((RCI_vertices, V[i]@yRCI_new))

    @jax.jit
    def get_parameter_samples(carry,multiplier,vertices):
        xu = vertices@multiplier
        x = xu[0:nx]
        u = xu[nx:nx+nu]
        p = parameter_fcn(x,u,Win_new,bin_new,Whid_new,bhid_new,Wout_new,bout_new)
        return carry, p
    
    par_parameter_samples = partial(get_parameter_samples, vertices = RCI_vertices.T)
    num_samples= 10000
    multiplier_samples = np.random.dirichlet(np.ones(m_bar), size=num_samples)
    p_samples = jax.lax.scan(par_parameter_samples, jnp.zeros((0,)), multiplier_samples)[1]

    for i in range(nq):
        color = plt.cm.viridis(i/nq)
        plt.plot(p_samples[:,i], color = color)
        plt.plot(a_new_new[i]*np.ones(num_samples), color = color, linestyle = '--')
    plt.show()
    p_min = np.min(p_samples, axis=0)
    p_max = np.max(p_samples, axis=0)
    print('p_min:', p_min)
    print('a_new:', a_new_new)

    if nx<=3:
        RCI_set = Polytope(A = F, b = yRCI_new)
        RCI_set.plot()
        plt.show()
    
    #Check BFRs
    model_iLPV_BFR = {'A': A, 'B': B, 'L': L, 'C': C, 'Win': Win, 'bin': bin, 'Whid': Whid, 'bhid': bhid, 'Wout': Wout, 'bout': bout}
    BFR_train_qLPV_CL, y_train_qLPV_CL, x0_train_qLPV_CL = qLPV_BFR(model_LPV_concur, Us_train, Ys_train, observer = True, activation_func = activation_func)
    BFR_train_qLPV_OL, y_train_qLPV_OL, x0_train_qLPV_OL = qLPV_BFR(model_LPV_concur, Us_train, Ys_train, observer = False, activation_func = activation_func)
    BFR_train_iLPV, y_train_iLPV, x0_train_iLPV = qLPV_BFR(model_iLPV_BFR, Us_train, Ys_train, activation_func = activation_func)
    print('BFR train: qLPV OL', BFR_train_qLPV_OL, 'qLPV CL', BFR_train_qLPV_CL, 'init LPV', BFR_train_iLPV)

    BFR_observer_qLPV_CL, y_observer_qLPV_CL, x0_observer_qLPV_CL = qLPV_BFR(model_LPV_concur, [Us_observer], [Ys_observer], observer = True, activation_func = activation_func)
    BFR_observer_qLPV_OL, y_observer_qLPV_OL, x0_observer_qLPV_OL = qLPV_BFR(model_LPV_concur, [Us_observer], [Ys_observer], observer = False, activation_func = activation_func)
    BFR_observer_iLPV, y_observer_iLPV, x0_observer_iLPV = qLPV_BFR(model_iLPV_BFR, [Us_observer], [Ys_observer], activation_func = activation_func)
    print('BFR observer: qLPV OL', BFR_observer_qLPV_OL, 'qLPV CL', BFR_observer_qLPV_CL, 'init LPV', BFR_observer_iLPV)

    BFR_test_qLPV_CL, y_test_qLPV_CL, x0_test_qLPV_CL = qLPV_BFR(model_LPV_concur, [Us_test], [Ys_test], observer = True, activation_func = activation_func)
    BFR_test_qLPV_OL, y_test_qLPV_OL, x0_test_qLPV_OL = qLPV_BFR(model_LPV_concur, [Us_test], [Ys_test], observer = False, activation_func = activation_func)
    BFR_test_iLPV, y_test_iLPV, x0_test_iLPV = qLPV_BFR(model_iLPV_BFR, [Us_test], [Ys_test], activation_func = activation_func)
    print('BFR test: qLPV OL', BFR_test_qLPV_OL, 'qLPV CL', BFR_test_qLPV_CL, 'init LPV', BFR_test_iLPV)

    
    #Save sim data
    model_LPV_concur['yhat_train_CL'] = y_train_qLPV_CL
    model_LPV_concur['yhat_test_CL'] = np.array(y_test_qLPV_CL[0])
    model_LPV_concur['yhat_train_OL'] = y_train_qLPV_OL[0]
    model_LPV_concur['yhat_test_OL'] = np.array(y_test_qLPV_OL[0])
    model_LPV_concur['yhat_observer_CL'] = np.array(y_observer_qLPV_CL[0])
    model_LPV_concur['yhat_observer_OL'] = np.array(y_observer_qLPV_OL[0])

    model_LPV_concur['BFR_train_qLPV_CL'] = np.array(BFR_train_qLPV_CL)
    model_LPV_concur['BFR_train_qLPV_OL'] = np.array(BFR_train_qLPV_OL)
    model_LPV_concur['BFR_train_iLPV'] = np.array(BFR_train_iLPV)
    model_LPV_concur['BFR_observer_qLPV_CL'] = np.array(BFR_observer_qLPV_CL)
    model_LPV_concur['BFR_observer_qLPV_OL'] = np.array(BFR_observer_qLPV_OL)
    model_LPV_concur['BFR_observer_iLPV'] = np.array(BFR_observer_iLPV)
    model_LPV_concur['BFR_test_qLPV_CL'] = np.array(BFR_test_qLPV_CL)
    model_LPV_concur['BFR_test_qLPV_OL'] = np.array(BFR_test_qLPV_OL)
    model_LPV_concur['BFR_test_iLPV'] = np.array(BFR_test_iLPV)

    return model_LPV_concur, RCI_concur




    



