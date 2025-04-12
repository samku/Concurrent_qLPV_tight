from imports import *

def qLPV_BFR(model_qLPV, input, output, observer = False, activation_func = 1):
    #This function only accepts list of input and outputs
    
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

    A = model_qLPV['A'].copy()
    B = model_qLPV['B'].copy() 
    C = model_qLPV['C'].copy()
    if observer:
        L = model_qLPV['L'].copy()
    Win = model_qLPV['Win'].copy()
    bin = model_qLPV['bin'].copy()
    Whid = model_qLPV['Whid'].copy()
    bhid = model_qLPV['bhid'].copy()
    Wout = model_qLPV['Wout'].copy()
    bout = model_qLPV['bout'].copy()

    #Extract sizes
    nq = A.shape[0]
    nx = A.shape[1]
    nu = B.shape[2]
    nu_original = nu
    ny = C.shape[0]
    nH = Whid.shape[1]+1

    #Adjust for observer
    if observer:
        nu_original = nu
        nu = nu+ny
        B_obsv = np.zeros((nq,nx,nu))
        for i in range(nq):
            A[i] = ((A[i])-(L[i])@C).copy()
            B_obsv[i] = np.hstack((B[i],L[i])).copy()
        B = B_obsv.copy()

    #Simulation functions
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
    def state_fcn(x,u):
        p = parameter_fcn(x,u[:nu_original])
        A_tot = jnp.zeros((nx,nx))
        B_tot = jnp.zeros((nx,nu))
        for i in range(nq):
            A_tot += A[i]*p[i]
            B_tot += B[i]*p[i]
        return A_tot@x+B_tot@u

    @jax.jit
    def output_fcn(x):
        return C@x
    
    @jax.jit
    def SS_forward(x, u):
        y_current = output_fcn(x)
        x_next = state_fcn(x, u).reshape(-1)
        return x_next, y_current
    
    #Optimize initial state
    x0s_optimized = []
    simulator = partial(SS_forward)  
    num_datasets = len(input)
    BFRs = np.zeros(num_datasets)
    y_sims = []
    for idx in range(num_datasets):
        if observer == False:
            input_current = np.array(input[idx])
        else:
            input_current = np.hstack((input[idx], output[idx]))
        def predict_x0(x0):
            y_sim = jax.lax.scan(simulator, x0, input_current)[1]
            return jnp.sum((output[idx]-y_sim)**2)
        options_BFGS = lbfgs_options(iprint=5, iters=1000, lbfgs_tol=1.e-10, memory=100)
        options_BFGS['disp'] = True
        solver = jaxopt.ScipyBoundedMinimize(
            fun=predict_x0, tol=1.e-10, method="L-BFGS-B", maxiter=1000, options=options_BFGS)
        x0_optimized, state = solver.run(jnp.zeros((nx,)), bounds=(-100*np.ones(nx), 100*np.ones(nx)))  
        
        #Simulate with optimized initial state
        x0_optimized = jnp.array(x0_optimized)
        y_sim = jax.lax.scan(simulator, x0_optimized, input_current)[1]

        numerator = np.sum((output[idx] - y_sim) ** 2)
        denominator = np.sum((output[idx] - np.mean(output[idx])) ** 2)+1.e-10
        BFR = np.maximum(0, 1 - np.sqrt(numerator/denominator))*100
        BFRs[idx] = BFR
        y_sims.append(y_sim)
        x0s_optimized.append(x0_optimized)

    return BFRs, y_sims, x0s_optimized
 