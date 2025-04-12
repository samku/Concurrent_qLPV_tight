import numpy as np
from plant_models.plant_model import Dynamics
from scipy.integrate import solve_ivp

m = 1.5
d = 1.
a = 1.
b = 2000.

class oscillator(Dynamics):
    def __init__(self, initial_state, 
                dt = 0.1, m = m, d = d, a = a, b = b,
                qx = 0.0005, qy = 0.0005, 
                u_lim = 0.5, type='DT'):
        
        super().__init__(initial_state)
        self.qx = qx
        self.qy = qy
        self.u_lim = u_lim
        self.type = type
        self.dt = dt
        self.params = [m, d, a, b]
        self.nx_plant = 2
        self.nu_plant = 1
        self.ny_plant = 1
        self.C = np.array([[1,0]])

    def dynamics(self, x, u):
        f = lambda t, x: silverbox_dynamics(x, u, self.params)
        sol = solve_ivp(f, [0, self.dt], x, method='RK45', rtol=1.e-8, atol=1.e-8)
        return sol.y[:, -1] + self.qx*np.random.randn(self.nx_plant)
    
    def output(self, x, u):
        y = self.C@x + self.qy*np.random.randn(self.ny_plant)
        return np.array([y[0]])

def silverbox_dynamics(x, u, params):
    m, d, a, b = params
    dx0 = x[1]
    dx1 = (1/m) * (u[0] - d * x[1] - a * x[0] - b * x[0]**3)
    return np.array([dx0, dx1])
    
    