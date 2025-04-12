from imports import *

def predict_initial_states(model_qLPV, input, output, activation=1):

    M = len(input)
    N = len(input[0]) #Assume all trajectories are of the same length

    from qLPV_jax import qLPV_jax
    model_qLPV_jax = qLPV_jax(model_qLPV.copy(), activation=activation)
    nx = model_qLPV_jax.nx
    nu = model_qLPV_jax.nu
    ny = model_qLPV_jax.ny

    #Get initial states for each trajectory
    x0_chunks = np.zeros((M,nx))
    iters = 100000
    maxiter = 100000
    memory = 5
    jax.config.update("jax_enable_x64", True)
    @jax.jit
    def predict_x0_single(x0, z_current):
        u_current = z_current[:, :nu]
        y_current = z_current[:, nu:]
        x_sim, y_sim = model_qLPV_jax.simulator(x0, u_current)
        return jnp.sum((y_current-y_sim)**2)
    
    @jax.jit
    def predict_x0_parallel(carry, xz_list):
        x0 = xz_list[0]
        z = xz_list[1]
        return carry, predict_x0_single(x0, z)
    par_predict_x0 = partial(predict_x0_parallel)

    z_all = np.zeros((M, N, nu+ny))
    for idx in range(M): 
        z_all[idx] = np.hstack((input[idx], output[idx]))
    
    @jax.jit
    def total_x0_cost_chunks(x0):
        costs = jax.lax.scan(par_predict_x0, 0., [x0, z_all])[1]
        return jnp.sum(costs)
    
    options_BFGS = lbfgs_options(iprint=5, iters=iters, lbfgs_tol=1.e-10, memory=memory)
    jax.config.update("jax_enable_x64", True)
    solver = jaxopt.ScipyBoundedMinimize(
            fun=total_x0_cost_chunks, tol=1.e-10, method="L-BFGS-B", maxiter=maxiter, options=options_BFGS)
    x0_chunks, state = solver.run(jnp.zeros((M,nx)), bounds=(-100*np.ones((M,nx)), 100*np.ones((M,nx))))  
    x0_chunks = np.array(x0_chunks)

    #Check BFR
    for idx in range(M):
        x_sim, y_sim = model_qLPV_jax.simulator(x0_chunks[idx], input[idx])
        numerator = np.sum((output[idx] - y_sim) ** 2)
        denominator = np.sum((output[idx] - np.mean(output[idx])) ** 2)+1.e-10
        BFR_local = np.maximum(0, 1 - np.sqrt(numerator/denominator))*100
        print('Chunk:',idx, 'of', M, 'has BFR:',BFR_local)

    return x0_chunks