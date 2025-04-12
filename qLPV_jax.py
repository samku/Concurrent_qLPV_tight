from imports import *

class qLPV_jax:
    def __init__(self, model_qLPV, activation=1):

        if activation == 1:
            activation = nn.swish
        elif activation == 2:
            activation = nn.relu
        elif activation == 3:
            @jax.jit
            def activation(x):
                return nn.elu(x)+1.
        elif activation == 4:
            activation = nn.sigmoid
            

        self.activation = activation
        self.A = model_qLPV['A'].copy()
        self.B = model_qLPV['B'].copy() 
        self.C = model_qLPV['C'].copy()
        self.L = model_qLPV['L'].copy()
        self.Win = model_qLPV['Win'].copy()
        self.bin = model_qLPV['bin'].copy()
        self.Whid = model_qLPV['Whid'].copy()
        self.bhid = model_qLPV['bhid'].copy()
        self.Wout = model_qLPV['Wout'].copy()
        self.bout = model_qLPV['bout'].copy()

        #Extract sizes
        self.nq = self.A.shape[0]
        self.nx = self.A.shape[1]
        self.nu = self.B.shape[2]
        self.ny = self.C.shape[0]
        self.nth = self.Win.shape[1]
        self.nH = self.Whid.shape[1]+1
    
    def simulator(self, x0, u):
        @jax.jit
        def parameter_fcn(x,u):
            z = x
            p = jnp.zeros((self.nq,))
            for i in range(self.nq):
                post_linear = self.Win[i]@z+self.bin[i]
                post_activation = self.activation(post_linear)
                for j in range(self.nH-1): 
                    post_linear = self.Whid[i][j]@post_activation+self.bhid[i][j]
                    post_activation = self.activation(post_linear)
                post_linear = self.Wout[i]@post_activation+self.bout[i]
                p = p.at[i].set(post_linear)
            p = jnp.exp(p)/jnp.sum(jnp.exp(p))
            return p
            
        @jax.jit
        def state_fcn(x, u):
            p = parameter_fcn(x, u)
            A_tot = jnp.zeros((self.nx,self.nx))
            B_tot = jnp.zeros((self.nx,self.nu))
            for i in range(self.nq):
                A_tot += self.A[i]*p[i]
                B_tot += self.B[i]*p[i]
            return A_tot@x + B_tot@u

        @jax.jit
        def output_fcn(x):
            return self.C@x

        @jax.jit
        def SS_forward(x, u):
            y_current = output_fcn(x)
            x_next = state_fcn(x, u).reshape(-1)
            return x_next, jnp.hstack((x_next, y_current))
    
        simulator_partial = partial(SS_forward)
        xy_sim = jax.lax.scan(simulator_partial, x0, u)[1]
        x_sim = xy_sim[:,0:self.nx]
        y_sim = xy_sim[:,self.nx:]
        x_sim = jnp.vstack((x0, x_sim)) #Length of x_sim is T+1
        return x_sim, y_sim
    
    def simulator_obsv(self, x0, uy):
        @jax.jit
        def parameter_fcn(x,u):
            z = x
            p = jnp.zeros((self.nq,))
            for i in range(self.nq):
                post_linear = self.Win[i]@z+self.bin[i]
                post_activation = self.activation(post_linear)
                for j in range(self.nH-1): 
                    post_linear = self.Whid[i][j]@post_activation+self.bhid[i][j]
                    post_activation = self.activation(post_linear)
                post_linear = self.Wout[i]@post_activation+self.bout[i]
                p = p.at[i].set(post_linear)
            p = jnp.exp(p)/jnp.sum(jnp.exp(p))
            return p

        @jax.jit
        def state_fcn_obsv(x, z):
            p = parameter_fcn(x, z[:self.nu])
            A_tot = jnp.zeros((self.nx,self.nx))
            B_tot = jnp.zeros((self.nx,self.nu))
            L_tot = jnp.zeros((self.nx,self.ny))
            for i in range(self.nq):
                A_tot += self.A[i]*p[i]
                B_tot += self.B[i]*p[i]
                L_tot += self.L[i]*p[i]
            return (A_tot-L_tot@self.C)@x + jnp.hstack((B_tot, L_tot))@z

        @jax.jit
        def output_fcn(x):
            return self.C@x

        @jax.jit
        def SS_forward_obsv(x, z):
            y_current = output_fcn(x)
            x_next = state_fcn_obsv(x, z).reshape(-1)
            return x_next, jnp.hstack((x_next, y_current))
    
        simulator_obsv_partial = partial(SS_forward_obsv)
        xy_sim = jax.lax.scan(simulator_obsv_partial, x0, uy)[1]
        x_sim = xy_sim[:,0:self.nx]
        y_sim = xy_sim[:,self.nx:]
        x_sim = jnp.vstack((x0, x_sim)) #Length of x_sim is T+1
        return x_sim, y_sim
    
    