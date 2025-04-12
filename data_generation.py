from imports import *
from utils.generate_file_path import generate_file_path
from utils.multisine_inputs import generate_multisine

current_directory = Path(__file__).parent

def generate_dataset(system, N_train, N_pieces, N_batch, N_observer, N_test, scale_data, folder_name, file_name, overwrite_data):
    #Data generation
    file_path = generate_file_path(folder_name, file_name, current_directory)
    if not file_path.exists() or overwrite_data:
        print('Generating dataset...')
        nu = system.nu_plant
        
        #Training data
        N_train_total = N_train*N_pieces
        #U_tot = generate_multisine(nu, N_train_total, 0.1, 100, -system.u_lim, system.u_lim, 1000)
        U_tot = -system.u_lim + 2.*np.random.rand(N_train_total, system.nu_plant) * np.array(system.u_lim)
        Y_tot = system.simulate(U_tot)
        Y_tot = Y_tot[:-1,:]
        U_train = []
        Y_train = []
        for i in range(N_pieces):
            U_train.append(U_tot[i*N_train:(i+1)*N_train,:])
            Y_train.append(Y_tot[i*N_train:(i+1)*N_train,:])


        #Observer data
        system.state = np.zeros(system.nx_plant)
        U_tot = -system.u_lim + 2.*np.random.rand(N_observer, system.nu_plant) * np.array(system.u_lim)
        Y_tot = system.simulate(U_tot)
        Y_tot = Y_tot[:-1,:]
        U_observer = U_tot[:N_observer,:]
        Y_observer = Y_tot[:N_observer,:]

        #Testing data
        system.state = np.zeros(system.nx_plant)
        U_tot = -system.u_lim + 2.*np.random.rand(N_test, system.nu_plant) * np.array(system.u_lim)
        Y_tot = system.simulate(U_tot)
        Y_tot = Y_tot[:-1,:]
        U_test = U_tot[:N_test,:]
        Y_test = Y_tot[:N_test,:]
        ny = Y_train[0].shape[1]

        #Batch data for AL
        N_batch_total = N_train*N_batch
        system.state = np.zeros(system.nx_plant)
        U_tot = -system.u_lim + 2.*np.random.rand(N_batch_total, system.nu_plant) * np.array(system.u_lim)
        Y_tot = system.simulate(U_tot)
        Y_tot = Y_tot[:-1,:]
        U_batch = []
        Y_batch = []
        for i in range(N_batch):
            U_batch.append(U_tot[i*N_train:(i+1)*N_train,:])
            Y_batch.append(Y_tot[i*N_train:(i+1)*N_train,:])

        Y_stacked = np.vstack(Y_train)
        U_stacked = np.vstack(U_train)
        #Scale data
        if scale_data == True:
            _, ymean, ygain = standard_scale(Y_stacked)
            _, umean, ugain = standard_scale(U_stacked)
        else:
            ymean = np.zeros((ny))
            ygain = np.ones((ny))
            umean = np.zeros((nu))
            ugain = np.ones((nu))
        Ys_train = []
        Us_train = []
        Ys_batch = []
        Us_batch = []
        for i in range(N_pieces):
            Us_train.append((U_train[i]-umean)*ugain)
            Ys_train.append((Y_train[i]-ymean)*ygain)
            plt.plot(Ys_train[i])

        for i in range(N_batch):
            Us_batch.append((U_batch[i]-umean)*ugain)
            Ys_batch.append((Y_batch[i]-ymean)*ygain)
            
        Us_test = (U_test-umean)*ugain
        Ys_test = (Y_test-ymean)*ygain
        Us_observer = (U_observer-umean)*ugain
        Ys_observer = (Y_observer-ymean)*ygain

        plt.plot(Ys_observer,'r--')
        plt.plot(Ys_test,'g--')
        plt.show()

        #Build constraints as box of inputs and convex hull of outputs
        u_lim_model = (system.u_lim-umean)*ugain
        HU = np.vstack((np.eye(nu),-np.eye(nu)))
        hU = np.hstack((u_lim_model,u_lim_model))
        Ys_stacked = np.vstack(Ys_train)
        Y_hull = Polytope(V = Ys_stacked)
        HY = Y_hull.A
        hY = 0.6*Y_hull.b
    
        constraints = {'HU': HU, 'hU': hU, 'HY': HY, 'hY': hY}
        Y_vert = Polytope(A = HY, b = hY)

        #Build dataset
        dataset = {}
        dataset['system'] = system
        dataset['U_train'] = U_train
        dataset['Y_train'] = Y_train
        dataset['U_test'] = U_test
        dataset['Y_test'] = Y_test
        dataset['u_scaler'] = np.array([umean, ugain])
        dataset['y_scaler'] = np.array([ymean, ygain])
        dataset['Us_train'] = Us_train
        dataset['Ys_train'] = Ys_train
        dataset['Us_batch'] = Us_batch
        dataset['Ys_batch'] = Ys_batch
        dataset['Us_test'] = Us_test
        dataset['Ys_test'] = Ys_test
        dataset['Us_observer'] = Us_observer
        dataset['Ys_observer'] = Ys_observer
        dataset['constraints'] = constraints

        #Save dataset
        with open(file_path, 'wb') as f:
            pickle.dump(dataset, f)
        print('Dataset saved to ', file_path)
    else:
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)

    return dataset
    
