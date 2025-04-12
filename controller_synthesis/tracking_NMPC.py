import sys
import os
base_dir = os.path.dirname(os.path.dirname(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
from imports import *
from qLPV_model import qLPV_model

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


#Extract RCI params
F = RCI_concur['F']
E = RCI_concur['E']
V = RCI_concur['V']
yRCI = np.array(RCI_concur['yRCI'])
uRCI = np.array(RCI_concur['uRCI'])
m_bar = len(V)
m = len(F)
w_hat = RCI_concur['W'][0]
epsw = RCI_concur['W'][1]
kappa = models['kappa']

#Extract parameters
A = model_LPV_concur['A']
B = model_LPV_concur['B']
C = model_LPV_concur['C']
L = model_LPV_concur['L']
Win = model_LPV_concur['Win']
bin = model_LPV_concur['bin']
Whid = model_LPV_concur['Whid']
bhid = model_LPV_concur['bhid']
Wout = model_LPV_concur['Wout']
bout = model_LPV_concur['bout']
nq = A.shape[0]
nu = B.shape[2]
nx = A.shape[1]
ny = C.shape[0]
sizes = models['sizes']
nH = sizes[-1]
activation = models['activation']
HU = dataset['constraints']['HU']
hU = dataset['constraints']['hU']
HY = dataset['constraints']['HY']
hY = dataset['constraints']['hY']
u_scaler = dataset['u_scaler']
y_scaler = dataset['y_scaler']


if activation == 1:
    activation_func = nn.swish
elif activation == 2:
    activation_func = nn.relu
elif activation == 3:
    @jax.jit
    def activation_func(x):
        return nn.elu(x)+1.
elif activation == 4:
    activation_func = nn.sigmoid

#Recompute matrices based on BP for controller
RCI_vertices = np.zeros((0,nx))
for i in range(m_bar):
    RCI_vertices = np.vstack((RCI_vertices, V[i]@yRCI))

max_RCI_vertices = np.max(RCI_vertices,axis=0)
min_RCI_vertices = np.min(RCI_vertices,axis=0)
mean_input = 0.5*(max_RCI_vertices+min_RCI_vertices)
var_input = 0.5*(max_RCI_vertices-min_RCI_vertices)
        
lu_before = np.zeros((nq, 2))
for i in range(nq):
    lb_curr = Win[i]@mean_input+bin[i]-np.abs(Win[i])@var_input
    ub_curr = Win[i]@mean_input+bin[i]+np.abs(Win[i])@var_input
    lb_activated = activation_func(lb_curr)
    ub_activated = activation_func(ub_curr)
    mean_lb_activated = 0.5*(lb_activated+ub_activated)
    var_lb_activated = 0.5*(ub_activated-lb_activated)
    for j in range(nH-1):     
        lb_curr = Whid[i][j]@mean_lb_activated+bhid[i][j]-np.abs(Whid[i][j])@var_lb_activated
        ub_curr = Whid[i][j]@mean_lb_activated+bhid[i][j]+np.abs(Whid[i][j])@var_lb_activated
        lb_activated = activation_func(lb_curr)
        ub_activated = activation_func(ub_curr)
        mean_lb_activated = 0.5*(lb_activated+ub_activated)
        var_lb_activated = 0.5*(ub_activated-lb_activated)
    lb_curr = Wout[i]@mean_lb_activated+bout[i]-np.abs(Wout[i])@var_lb_activated
    ub_curr = Wout[i]@mean_lb_activated+bout[i]+np.abs(Wout[i])@var_lb_activated
    lu_before[i] = np.hstack((np.exp(lb_curr), np.exp(ub_curr)))

a_bound = np.zeros((nq,))
for i in range(nq):
    numerator = lu_before[i,0]
    denominator = lu_before[i,0]
    for j in range(nq):
        if i!=j:
            denominator = denominator+lu_before[j,1]
    a_bound[i] = numerator/denominator

A_vertices = np.zeros((nq,nx,nx))
B_vertices = np.zeros((nq,nx,nu))
L_vertices = np.zeros((nq,nx,ny))
for i in range(nq):
    A_loc = (1-np.sum(a_bound))*A[i]
    B_loc = (1-jnp.sum(a_bound))*B[i]
    L_loc = (1-jnp.sum(a_bound))*L[i]
    for j in range(nq):
        A_loc = A_loc + a_bound[j]*A[j]
        B_loc = B_loc + a_bound[j]*B[j]
        L_loc = L_loc + a_bound[j]*L[j]
    A_vertices[i] = A_loc
    B_vertices[i] = B_loc
    L_vertices[i] = L_loc

#Build model dict for simulation
model_dict = {}
model_dict['A'] = A
model_dict['B'] = B
model_dict['C'] = C
model_dict['L'] = L
model_dict['Win'] = Win
model_dict['bin'] = bin
model_dict['Whid'] = Whid
model_dict['bhid'] = bhid
model_dict['Wout'] = Wout
model_dict['bout'] = bout
model_dict['sizes'] = sizes
model_dict['activation'] = activation
model_dict['u_scaler'] = u_scaler
model_dict['y_scaler'] = y_scaler
model_dict['HU'] = HU
model_dict['hU'] = hU
model_dict['HY'] = HY
model_dict['hY'] = hY
model_dict['W'] = np.vstack((w_hat, epsw))
model = qLPV_model(model_dict, activation = activation)

#Compute tracking bounds
tracking_output = 1
C = C[0] #Hardcoded for single output
xRCI_vert = np.zeros((m_bar, nx))
xRCI_vert_LTI = np.zeros((m_bar, nx))
xRCI_vert_LPV_init = np.zeros((m_bar, nx))
yRCI_vert = np.zeros((m_bar, ny))
yRCI_vert_LTI = np.zeros((m_bar, ny))
yRCI_vert_LPV_init = np.zeros((m_bar, ny))
for k in range(m_bar):
    xRCI_vert[k] = V[k] @ yRCI
    yRCI_vert[k] = C @ V[k] @ yRCI

y_track_max = model.y_scaler[0] + np.max(yRCI_vert)/model.y_scaler[1]
y_track_min = model.y_scaler[0] + np.min(yRCI_vert)/model.y_scaler[1]

Hcon_plant = HY * model.y_scaler[1]
hcon_plant = hY + Hcon_plant @ model.y_scaler[0]
Hcon = Polytope(A=Hcon_plant, b=hcon_plant)
Y_con_vert_plant = Hcon.V
y_max_con = np.max(Y_con_vert_plant, axis = 0)
y_min_con = np.min(Y_con_vert_plant, axis = 0)

#Compute input bounds
u_bounds_plant = model.u_scaler[0] + model.hU[0:model.nu]/model.u_scaler[1]

#Disturbance bounds
hW = np.array([model.W[0]+model.W[1], model.W[0]-model.W[1]]).reshape(1,-1)[0]
w_bounds_ub = model.y_scaler[0] + (model.W[0]+model.W[1])/model.y_scaler[1]
w_bounds_lb = model.y_scaler[0] + (model.W[0]-model.W[1])/model.y_scaler[1]

opts = {"verbose": False,  
        "ipopt.print_level": 0,  
        "print_time": 0 }

#Simulate in closed loop
N_sim = 200
N_MPC = 1

x_plant = np.zeros((N_sim+1,system.nx_plant))
system.state = x_plant[0]
u_plant = np.zeros((N_sim,system.nu_plant))
y_plant = np.zeros((N_sim,model.ny))
parameters_sim = np.zeros((N_sim, model.nq))

x_model = np.zeros((N_sim+1,model.nx))
x_next_bad = np.zeros((N_sim+1,model.nx))
y_model = np.zeros((N_sim,model.ny))
p_model = np.zeros((N_sim,model.nq))
dy_model = np.zeros((N_sim,model.ny))

ref_y = np.zeros((N_sim, 1))

for t in range(N_sim):
    #print(t)
    y_model[t] = model.output(x_model[t])
    y_plant[t] = system.output(x_plant[t], 0.)
    dy_model[t] = y_plant[t] - y_model[t]

    if t%30 == 0 or t == 0:
        #Reference in plant space
        ref_y[t] = 1.2*(y_track_min + (y_track_max-y_track_min)*np.random.rand(1))
    else:
        ref_y[t] = ref_y[t-1]
    if t<=30:
        ref_y[t] = y_track_min*1.1
    elif t>30 and t<=40:
        ref_y[t] = y_track_max*1.1
    
    #MPC
    if N_MPC>1:
        opti = ca.Opti()
        u_MPC = opti.variable(model.nu, N_MPC)
        x_MPC = opti.variable(model.nx, N_MPC+1)
        x_next_1 = model.dynamics(x_model[t], \
                                model.u_scaler[0] + u_MPC[:,0]/model.u_scaler[1], \
                                y_plant[t])
        cost = 0.
        for i in range(N_MPC):
            if i == 0:
                opti.subject_to(x_MPC[:,i] == x_model[t])
            elif i == 1:
                opti.subject_to(x_MPC[:,i] == x_next_1)
            else:
                opti.subject_to(x_MPC[:,i] == model.dynamics_OL(x_MPC[:,i-1],u_MPC[:,i-1]))
            vector = F @ x_MPC[:,i] - yRCI
            opti.subject_to(vector<=0)
            vector = model.HU @ u_MPC[:,i] - model.hU
            opti.subject_to(vector<=0)
            error = model.output(x_MPC[:,i]) - ref_y[t]
            cost += ca.dot(error,error)

        opti.minimize(cost)
        opti.solver('ipopt',opts)
        sol = opti.solve()
        x_MPC = sol.value(x_MPC).T
        u_MPC = sol.value(u_MPC)
        u_plant[t] = model.u_scaler[0] + (u_MPC[0].T)/model.u_scaler[1]
        parameters_sim[t] = model.parameter(x_model[t], u_MPC[0]).reshape(1,-1)
    else:
        opti = ca.Opti()
        u_MPC = opti.variable(model.nu, N_MPC)
        x_next_1 = model.dynamics(x_model[t], \
                                model.u_scaler[0] + u_MPC[:,0]/model.u_scaler[1], \
                                y_plant[t])
        vector = F @ x_next_1 - yRCI
        opti.subject_to(vector<=0)
        vector = model.HU @ u_MPC[:,0] - model.hU
        opti.subject_to(vector<=0)
        error = model.output(x_next_1) - ref_y[t]
        opti.minimize(ca.dot(error,error))
        opti.solver('ipopt',opts)
        sol = opti.solve()
        u_MPC = sol.value(u_MPC)
        u_plant[t] = model.u_scaler[0] + (u_MPC)/model.u_scaler[1]
        parameters_sim[t] = model.parameter(x_model[t], u_MPC).reshape(1,-1)

    

    #Propagate
    system.update(u_plant[t])
    x_plant[t+1] = system.state
    x_model[t+1] = model.dynamics(x_model[t],u_plant[t],y_plant[t])
    

import matplotlib.gridspec as gridspec

# Set up the figure and the custom grid layout
plt.rcParams['text.usetex'] = True
figure = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])  # Give ax3 more vertical space

# Create subplots with custom positions
ax1 = figure.add_subplot(gs[0])
ax2 = figure.add_subplot(gs[1])

# --- Plotting on ax1 ---
ax1.plot(np.arange(0,N_sim)*dt,ref_y,'k--')
ax1.plot(np.arange(0,N_sim)*dt,y_plant, 'g')
ax1.plot(np.arange(0,N_sim)*dt,y_model,'r--')
ax1.plot(np.arange(0,N_sim)*dt,y_max_con*np.ones(N_sim),'k')
ax1.plot(np.arange(0,N_sim)*dt,y_min_con*np.ones(N_sim),'k')
ax1.plot(np.arange(0,N_sim)*dt,y_track_max*np.ones(N_sim),'b:')
ax1.plot(np.arange(0,N_sim)*dt,y_track_min*np.ones(N_sim),'b:')
ax1.set_xlabel(r"$t$")
ax1.set_ylabel(r"$y$")
ax1.grid(True)
ax1.set_xlim([0, dt*(N_sim-1)]) 

# --- Plotting on ax3 ---
if nx == 2:
    Polytope(A=F, b=yRCI).plot(ax=ax2, patch_args = {"facecolor": 'r', "alpha": 0.8, "linewidth": 1, "linestyle": '-', "edgecolor": 'r'})
else:
    Polytope(A=F, b=yRCI).projection(project_away_dim = np.linspace(2,nx)).plot(ax=ax2, patch_args = {"facecolor": 'r', "alpha": 0.8, "linewidth": 1, "linestyle": '-', "edgecolor": 'k'})

ax2.plot(x_model[:,0], x_model[:,1], 'g', linewidth = 2)
ax2.scatter(x_model[0,0], x_model[0,1], color = 'g')
ax2.autoscale()
ax2.set_xlabel(r"$z_1$")
ax2.set_ylabel(r"$z_2$")
ax2.grid(True)

plt.tight_layout()
plt.show()

