from imports import *
current_directory = Path(__file__).parent
from utils.generate_file_path import generate_file_path
from predict_initial_states import predict_initial_states
from data_generation import generate_dataset
from initialization_qLPV import initialization
from concurrent_identification import concurrent_identification
from sequential_benchmark import sequential_benchmark

#Generate/Load dataset
from plant_models.oscillator import oscillator
np.random.seed(21)

system = oscillator(initial_state=np.zeros(2))
overwrite_data = False
scale_data = True
N_train = 10000
N_pieces = 4
N_batch = 1
N_observer = 2000
N_test = 10000

folder_name = 'oscillator'
file_name = 'dataset'
dataset = generate_dataset(system, N_train, N_pieces, N_batch, N_observer, N_test, scale_data, folder_name, file_name, overwrite_data) 

file_name = 'initial_models'
overwrite_data = False
file_path = generate_file_path(folder_name, file_name, current_directory)
if not file_path.exists() or overwrite_data:
    nx = 2 #State dimension
    nq = 6 #Number of parameters
    nth = 3 #Number of neurons in each layer
    nH = 1 #Number of activation layers
    sizes = (nx, nq, nth, nH)
    kappa = 1.01 #Disturbance set inflation factor
    N_MPC = 5 #Horizon length for RCI set size factor
    activation = 3 #Activation function for qLPV 1-swish, 2-relu, 3-elu, 4-sigmoid

    id_params = {'eta': 0.001, 'rho_th': 0.0000, 'adam_epochs': 250, 'lbfgs_epochs': 0, 'iprint': 100, 'memory': 100,
                'train_x0': True, 'weight_RCI':1., 'N_MPC': N_MPC, 'kappa_p': 0., 'kappa_x': 0.01}
    model_LPV, RCI_LPV = initialization(dataset, sizes, kappa, id_params, activation_func = activation)
    models = {}
    models['model_LPV'] = model_LPV
    models['RCI_LPV'] = RCI_LPV
    models['sizes'] = sizes
    models['kappa'] = kappa
    models['N_MPC'] = N_MPC
    models['activation'] = activation

    #Save dataset
    with open(file_path, 'wb') as f:
        pickle.dump(models, f)
    print('Initial models saved to ', file_path)
else:
    with open(file_path, 'rb') as f:
        models = pickle.load(f)

#Concurrent identification
file_name = 'new_0.07'
overwrite_data = True
file_path = generate_file_path(folder_name, file_name, current_directory)
if not file_path.exists() or overwrite_data:
    id_params = {'eta': 0.001, 'rho_th': 0., 'rho_a':0.00, 'adam_epochs': 60000,'train_x0': True, 'weight_RCI': 0.0005, 'weight_LTI': 0., 'regularization_QP': 0.0005}
    model_LPV_concur, RCI_concur = concurrent_identification(dataset, models, id_params, use_bounds = True, zeta = 0.07, tightened_W = False, k_max =0)
    models_concur = {}
    models_concur['model_LPV_concur'] = model_LPV_concur
    models_concur['RCI_concur'] = RCI_concur
    models_concur['sizes'] = models['sizes']
    models_concur['kappa'] = models['kappa']
    models_concur['N_MPC'] = models['N_MPC']
    models_concur['activation'] = models['activation']

    #Save modelset
    with open(file_path, 'wb') as f:
        pickle.dump(models_concur, f)
    print('Concurrent identification saved to ', file_path)
else:
    with open(file_path, 'rb') as f:
        models_concur = pickle.load(f)

#model based RCI identification
file_name = 'model_based_RCI'
overwrite_data = False
file_path = generate_file_path(folder_name, file_name, current_directory)
if not file_path.exists() or overwrite_data:
    id_params = {'eta': 0.001, 'rho_th': 0.0001, 'adam_epochs': 2000, 'lbfgs_epochs': 2000, 'kappa_x': 0.01}
    model_LPV_concur, RCI_concur = sequential_benchmark(dataset, models, id_params)
    models_concur = {}
    models_concur['model_LPV_concur'] = model_LPV_concur
    models_concur['RCI_concur'] = RCI_concur
    models_concur['sizes'] = models['sizes']
    models_concur['kappa'] = models['kappa']
    models_concur['N_MPC'] = models['N_MPC']
    models_concur['activation'] = models['activation']

    #Save modelset
    with open(file_path, 'wb') as f:
        pickle.dump(models_concur, f)
    print('Concurrent identification saved to ', file_path)
else:
    with open(file_path, 'rb') as f:
        models_concur = pickle.load(f)










