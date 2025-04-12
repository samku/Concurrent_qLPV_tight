from imports import *
"""
Model class for qLPV systems for control design : plant input to plant output. Scaling handled inside
"""

def swish(x):
    return x/(1+np.exp(-x))

def swish_casadi(x):
    return x / (1 + ca.exp(-x))

def softplus(x):
    return np.log(1+np.exp(x))

def softplus_casadi(x):
    return ca.log(1 + ca.exp(x))

def elu(x):
    return np.where(x > 0, x, np.exp(x) - 1)+1.

def elu_casadi(x):
    return ca.if_else(x > 0, x, ca.exp(x) - 1)+1.

def relu(x):
    return np.maximum(0, x)

def relu_casadi(x):
    return ca.if_else(x > 0, x, 0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_casadi(x):
    return 1/(1+ca.exp(-x))

class qLPV_model:
    def __init__(self, parameters, activation = 1):
        self.sizes = parameters['sizes']
    
        self.activation = activation
        self.A = parameters['A']
        self.B = parameters['B']
        self.C = parameters['C']
        self.L = parameters['L']
        self.ny = self.C.shape[0]
        self.nu = self.B[0].shape[1]
        self.nx = self.sizes[0]
        self.nq = self.sizes[1]
        self.nth = self.sizes[2]
        self.nH = self.sizes[3]

        self.Win = parameters['Win']
        self.bin = parameters['bin']
        self.Whid = parameters['Whid']
        self.bhid = parameters['bhid']
        self.Wout = parameters['Wout']
        self.bout = parameters['bout']
        self.W = parameters['W']
        self.u_scaler = parameters['u_scaler']
        self.y_scaler = parameters['y_scaler']

        self.HU = parameters['HU']
        self.hU = parameters['hU']
        self.HY = parameters['HY']
        self.hY = parameters['hY']

    def scale_model(self, y_plant, u_plant):
        y_scaler = self.y_scaler
        u_scaler = self.u_scaler
        y_plant_scale = (y_plant-y_scaler[0])*y_scaler[1]
        u_plant_scale = (u_plant-u_scaler[0])*u_scaler[1]
        return y_plant_scale, u_plant_scale

    def parameter(self, x, u):
        if self.activation == 1:
            activation = swish
        elif self.activation == 2:
            activation = relu
        elif self.activation == 3:
            activation = elu
        elif self.activation == 4:
            activation = sigmoid

        z = x
        Win = self.Win
        bin = self.bin
        Whid = self.Whid
        bhid = self.bhid
        Wout = self.Wout
        bout = self.bout
        nq = self.nq
        nH = self.nH
        p = np.zeros((nq,))
        for i in range(nq):
            post_linear = Win[i]@z+bin[i]
            post_activation = activation(post_linear)
            for j in range(nH-1): 
                post_linear = Whid[i][j]@post_activation+bhid[i][j]
                post_activation = activation(post_linear)
            post_linear = Wout[i]@post_activation+bout[i]
            p[i] = np.exp(post_linear)
        p = p/np.sum(p)
        return p

    def parameter_casadi(self, x, u):
        if self.activation == 1:
            activation = swish_casadi
        elif self.activation == 2:
            activation = relu_casadi
        elif self.activation == 3:
            activation = elu_casadi
        elif self.activation == 4:
            activation = sigmoid_casadi
            
        z = x
        Win = self.Win
        bin = self.bin
        Whid = self.Whid
        bhid = self.bhid
        Wout = self.Wout
        bout = self.bout
        nq = self.nq
        nH = self.nH
        p = ca.DM.ones(0)
        for i in range(nq):
            post_linear = Win[i] @ z + bin[i]
            post_activation = activation(post_linear)
            for j in range(nH-1): 
                post_linear = Whid[i][j] @ post_activation + bhid[i][j]
                post_activation = activation(post_linear)
            post_linear = ca.reshape(Wout[i], (1, -1)) @ post_activation + bout[i]
            p = ca.vertcat(p, ca.exp(post_linear))  # Append the exponential of the output
        p = p / ca.sum1(p)
        return p
    
    def observer(self, x, p, u_plant, y_plant):
        y, u = self.scale_model(y_plant, u_plant)
        C = self.C
        L = self.L
        nq = self.nq
        L_tot = L[0]*p[0]
        for i in range(1,nq):
            L_tot += L[i]*p[i]  
        return L_tot@(y-C@x)

    def dynamics(self, x, u_plant, y_plant):
        y, u = self.scale_model(y_plant, u_plant)
        if isinstance(u_plant, ca.MX):
            #When using in optimization problems
            p = self.parameter_casadi(x, u)
        else:
            p = self.parameter(x, u)
        A = self.A
        B = self.B
        nq = self.nq
        A_tot = A[0]*p[0]
        B_tot = B[0]*p[0]
        for i in range(1,nq):
            A_tot += A[i]*p[i]
            B_tot += B[i]*p[i]
        return A_tot@x+B_tot@u+self.observer(x, p, u_plant, y_plant)
    
    def dynamics_OL(self, x, u):
        #Purely for MPC
        p = self.parameter_casadi(x, u)
        A = self.A
        B = self.B
        nq = self.nq
        A_tot = A[0]*p[0]
        B_tot = B[0]*p[0]
        for i in range(1,nq):
            A_tot += A[i]*p[i]
            B_tot += B[i]*p[i]
        return A_tot@x+B_tot@u
    
    def output(self, x):
        C = self.C
        y_scaler = self.y_scaler
        return ((C@x)/y_scaler[1])+y_scaler[0]

