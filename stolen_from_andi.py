import numpy as np
import csv
import andi
from math import floor

def normalize(trajs):    
    ''' Normalizes trajectories by substracting average and dividing by sqrt of msd
    Arguments:
	- traj: trajectory or ensemble of trajectories to normalize. 
    - dimensions: dimension of the trajectory.
	return: normalized trajectory'''
    if len(trajs.shape) == 1:
        trajs = np.reshape(trajs, (1, trajs.shape[0]))
    trajs = trajs - trajs.mean(axis=1, keepdims=True)
    displacements = (trajs[:,1:] - trajs[:,:-1]).copy()    
    variance = np.std(displacements, axis=1)
    variance[variance == 0] = 1    
    new_trajs = np.cumsum((displacements.transpose()/variance).transpose(), axis = 1)
    return np.concatenate((np.zeros((new_trajs.shape[0], 1)), new_trajs), axis = 1)

AD = andi.andi_datasets()

def andi_dataset_2(N = 1000, max_T = 1000, min_T = 10,
                           tasks = [1, 2, 3],
                           dimensions = [1,2,3], models = [2],
                           load_dataset = False, save_dataset = False, path_datasets = '',
                           load_trajectories = False, save_trajectories = False, path_trajectories = 'datasets/',
                           N_save = 1000, t_save = 1000, noise = [0.1, 0.3, 1]):  
                    
        print('Creating a dataset for task(s) '+str(tasks)+' and dimension(s) '+str(dimensions)+'.')
        
        # Checking inputs for errors
        if isinstance(dimensions, int) or isinstance(dimensions, float):
            dimensions = [dimensions]
        if isinstance(tasks, int) or isinstance(tasks, float):
            tasks = [tasks]
        
        # Define return datasets
        X1 = [[],[],[]]; X2 = [[],[],[]]; X3 = [[],[],[]]
        Y1 = [[],[],[]]; Y2 = [[],[],[]]; Y3 = [[],[],[]]
        
        if load_dataset or save_dataset:
            # Define name of result files, if needed
            task1 = path_datasets+'task1.txt'; ref1 = path_datasets+'ref1.txt'
            task2 = path_datasets+'task2.txt'; ref2 = path_datasets+'ref2.txt'
            task3 = path_datasets+'task3.txt'; ref3 = path_datasets+'ref3.txt'
        
        # Loading the datasets if chosen.
        if load_dataset:            
            for idx, (task, lab) in enumerate(zip([task1, task2, task3], [ref1, ref2, ref3])):
                if idx+1 in tasks:
                    
                    try:
                        t = csv.reader(open(task,'r'), delimiter=';', 
                                        lineterminator='\n',quoting=csv.QUOTE_NONNUMERIC)
                        l = csv.reader(open(lab,'r'), delimiter=';', 
                                        lineterminator='\n',quoting=csv.QUOTE_NONNUMERIC)
                    except:
                        raise FileNotFoundError('File for task '+str(idx+1)+' not found.')
                    
                    for _, (trajs, labels) in enumerate(zip(t, l)):   
                        if task == task1:                            
                            X1[int(trajs[0])-1].append(trajs[1:])
                            Y1[int(trajs[0])-1].append(labels[1])
                        if task == task2:
                            X2[int(trajs[0])-1].append(trajs[1:])
                            Y2[int(trajs[0])-1].append(labels[1])
                        if task == task3:
                            X3[int(trajs[0])-1].append(trajs[1:])
                            Y3[int(trajs[0])-1].append(labels[1:]) 
                    # Checking that the dataset exists in the files
                    for dim in dimensions:
                        if task == task1 and X1[dim-1] == []:
                            raise FileNotFoundError('Dataset for dimension '+str(dim)+' not contained in file task1.txt.')
                        if task == task2 and X2[dim-1] == []:
                            raise FileNotFoundError('Dataset for dimension '+str(dim)+' not contained in file task2.txt.')
                        if task == task3 and X3[dim-1] == []:
                            raise FileNotFoundError('Dataset for dimension '+str(dim)+' not contained in file task3.txt.')
                        
            return X1, Y1, X2, Y2, X3, Y3        

            
        exponents_dataset = np.arange(0.05, 1.96, 0.05)
        
        # Define return datasets
        X1 = [[],[],[]]; X2 = [[],[],[]]; X3 = [[],[],[]]
        Y1 = [[],[],[]]; Y2 = [[],[],[]]; Y3 = [[],[],[]]
        print('A')
        # Initialize the files
        if save_dataset:
            if 1 in tasks:
                csv.writer(open(task1,'w'), delimiter=';', lineterminator='\n',)
                csv.writer(open(ref1,'w'), delimiter=';', lineterminator='\n',)
            elif 2 in tasks:
                csv.writer(open(task2,'w'), delimiter=';', lineterminator='\n',)
                csv.writer(open(ref2,'w'), delimiter=';',lineterminator='\n',)
            elif 3 in tasks:
                csv.writer(open(task3,'w'), delimiter=';', lineterminator='\n',)
                csv.writer(open(ref3,'w'), delimiter=';',lineterminator='\n',)
        
        for dim in [1,2,3]:             
            if dim not in dimensions:
                continue
        
            print('B')
        #%%  Generate the dataset of the given dimension
            print('Generating dataset for dimension '+str(dim)+'.')
            dataset = AD.create_dataset(T = max_T, N= floor(N / len(exponents_dataset)), exponents = exponents_dataset, 
                                           models = models, dimension = dim,
                                           load_trajectories = False, save_trajectories = False, N_save = 100,
                                           path = path_trajectories)            
            
            print('C')
        #%% Normalize trajectories
    
            trajs = normalize(dataset[:,2:].reshape(dataset.shape[0]*dim, max_T))
            dataset[:,2:] = trajs.reshape(dataset[:,2:].shape)
    
        #%% Add localization error, Gaussian noise with sigma = noise
            loc_error_amplitude = np.random.choice(np.array(noise), size = dataset.shape[0]*dim)
            loc_error = (np.random.randn(dataset.shape[0]*dim, int((dataset.shape[1]-2)/dim)).transpose()*loc_error_amplitude).transpose()
                        
            dataset = AD.create_noisy_localization_dataset(dataset.copy(), dimension = dim, T = max_T, noise_func = loc_error)
            print('D')
            #%% Add random diffusion coefficients
                
            trajs = dataset[:,2:].reshape(dataset.shape[0]*dim, max_T)
            displacements = trajs[:,1:] - trajs[:,:-1]
            # Get new diffusion coefficients and displacements
            diffusion_coefficients = np.random.randn(trajs.shape[0])
            new_displacements = (displacements.transpose()*diffusion_coefficients).transpose()  
            # Generate new trajectories and add to dataset
            new_trajs = np.cumsum(new_displacements, axis = 1)
            new_trajs = np.concatenate((np.zeros((new_trajs.shape[0], 1)), new_trajs), axis = 1)
            dataset[:,2:] = new_trajs.reshape(dataset[:,2:].shape)
            
        
        #%% Task 1 - Anomalous exponent
            if 1 in tasks:         
                # Creating semi-balanced datasets
                N_exp = int(np.ceil(1.1*N/len(exponents_dataset)))
                for exponent in exponents_dataset:
                    dataset_exp = dataset[dataset[:,1] == exponent].copy()
                    np.random.shuffle(dataset_exp)
                    dataset_exp = dataset_exp[:N_exp, :]
                    try:
                        dataset_1 = np.concatenate((dataset_1, dataset_exp), axis = 0) 
                    except:
                        dataset_1 = dataset_exp
                np.random.shuffle(dataset_1)
                dataset_1 = dataset_1[:N,:]               
                
                info_cut = []
                for traj in dataset_1[:N, :]:             
                    # Cutting trajectories
                    cut = np.random.randint(min_T, max_T)            
                    info_cut.append(cut)            
                    traj_cut = traj[2:].reshape(dim, max_T)
                    traj_cut = traj_cut[:, :cut].reshape(dim*cut).tolist()               
                    # Saving dataset
                    X1[dim-1].append(traj_cut)
                    Y1[dim-1].append(np.around(traj[1],2))
                    if save_dataset:                        
                        AD.save_row(np.append(dim, traj_cut), task1)
                        AD.save_row(np.append(dim, np.around([traj[1]],2)), ref1)
                        
def andi_dataset_3(N = 1000, max_T = 1000, min_T = 10,
                        tasks = [1, 2, 3],
                        dimensions = [1,2,3],
                        load_dataset = False, save_dataset = False, path_datasets = '',
                        load_trajectories = False, save_trajectories = False, path_trajectories = 'datasets/',
                        N_save = 1000, t_save = 1000, noise = [0.1, 0.3, 1]):  
    ''' Creates a dataset similar to the one given by in the ANDI challenge. 
    Check the webpage of the challenge for more details. The default values
    are similar to the ones used to generate the available dataset.
    Arguments:  
        :N (int, numpy.array):
            - if int, number of trajectories per class (i.e. exponent and model) in the dataset.
            - if numpy.array, number of trajectories per classes: size (number of models)x(number of classes)    
        :max_T (int):
            - Maximum length of the trajectories in the dataset.
        :min_T (int):
            - Minimum length of the trajectories in the dataset.
        :tasks (int, array):
            - Task(s) of the ANDI for which datasets will be generated. Task 1 corresponds to the
            anomalous exponent estimation, Task 2 to the model prediction and Task 3 to the segmen-
            tation problem.
        :dimensions (int, array):
            - Task(s) for which trajectories will be generated. Three possible values: 1, 2 and 3.
        :load_dataset (bool):
            - if True, the module loads the trajectories from the files task1.txt, task2.txt and
            task3.txt and the labels from ref1.txt, ref2.txt and ref3.txt. If the trajectories do
            not exist but the file does, the module returns and empty dataset.
        :save_dataset (bool):
            - if True, the module saves the datasets in a .txt following the format discussed in the 
            webpage of the comptetion.
        :load_trajectories (bool):
            - if True, the module loads the trajectories of an .h5 file.  
        :save_trajectories (bool):
            - if True, the module saves a .h5 file for each model considered, with N_save trajectories 
                and T = T_save..
        :N_save (int):
            - Number of trajectories to save for each exponents/model. Advise: save at the beggining
                a big dataset (i.e. with default t_save N_save) which allows you to load any other combiantion
                of T and N.
        :t_save (int):
            - Length of the trajectories to be saved. See comments on N_save.                
    Return:
        The function returns 6 variables, three variables for the trajectories and three 
        for the corresponding labels. Each variable is a list of three lists. Each of the
        three lists corresponds to a given dimension, in ascending order. If one of the
        tasks/dimensions was not calculated, the given list will be empty
        :X1 (list of three lists):
            - Trajectories corresponding to Task 1. 
        :Y1 (list of three lists):
            - Labels corresponding to Task 1
        :X2 (list of three lists):
            - Trajectories corresponding to Task 2. 
        :Y2 (list of three lists):
            - Labels corresponding to Task 2
        :X3 (list of three lists):
            - Trajectories corresponding to Task 3. 
        :Y3 (list of three lists):
            - Labels corresponding to Task 3      '''
                
    print('Creating a dataset for task(s) '+str(tasks)+' and dimension(s) '+str(dimensions)+'.')
    
    # Checking inputs for errors
    if isinstance(dimensions, int) or isinstance(dimensions, float):
        dimensions = [dimensions]
    if isinstance(tasks, int) or isinstance(tasks, float):
        tasks = [tasks]
    
    # Define return datasets
    X1 = [[],[],[]]; X2 = [[],[],[]]; X3 = [[],[],[]]
    Y1 = [[],[],[]]; Y2 = [[],[],[]]; Y3 = [[],[],[]]
    
    if load_dataset or save_dataset:
        # Define name of result files, if needed
        task1 = path_datasets+'task1.txt'; ref1 = path_datasets+'ref1.txt'
        task2 = path_datasets+'task2.txt'; ref2 = path_datasets+'ref2.txt'
        task3 = path_datasets+'task3.txt'; ref3 = path_datasets+'ref3.txt'
    
    # Loading the datasets if chosen.
    if load_dataset:            
        for idx, (task, lab) in enumerate(zip([task1, task2, task3], [ref1, ref2, ref3])):
            if idx+1 in tasks:
                
                try:
                    t = csv.reader(open(task,'r'), delimiter=';', 
                                    lineterminator='\n',quoting=csv.QUOTE_NONNUMERIC)
                    l = csv.reader(open(lab,'r'), delimiter=';', 
                                    lineterminator='\n',quoting=csv.QUOTE_NONNUMERIC)
                except:
                    raise FileNotFoundError('File for task '+str(idx+1)+' not found.')
                
                for idx2, (trajs, labels) in enumerate(zip(t, l)):   
                    if task == task1:                            
                        X1[int(trajs[0])-1].append(trajs[1:])
                        Y1[int(trajs[0])-1].append(labels[1])
                    if task == task2:
                        X2[int(trajs[0])-1].append(trajs[1:])
                        Y2[int(trajs[0])-1].append(labels[1])
                    if task == task3:
                        X3[int(trajs[0])-1].append(trajs[1:])
                        Y3[int(trajs[0])-1].append(labels[1:]) 
                # Checking that the dataset exists in the files
                for dim in dimensions:
                    if task == task1 and X1[dim-1] == []:
                        raise FileNotFoundError('Dataset for dimension '+str(dim)+' not contained in file task1.txt.')
                    if task == task2 and X2[dim-1] == []:
                        raise FileNotFoundError('Dataset for dimension '+str(dim)+' not contained in file task2.txt.')
                    if task == task3 and X3[dim-1] == []:
                        raise FileNotFoundError('Dataset for dimension '+str(dim)+' not contained in file task3.txt.')
                    
        return X1, Y1, X2, Y2, X3, Y3        

        
    exponents_dataset = np.arange(0.05, 2.01, 0.05)
    # Trajectories per model and exponent. Arbitrarely chose to fulfill balanced classes
    N_models = np.ceil(1.6*N/5)
    subdif = 20
    superdif = 21
    num_per_class =  np.zeros((len(AD.avail_models_name), len(exponents_dataset)))
    # ctrw, attm
    num_per_class[:2,:20] = np.ceil(N_models/subdif)
    # fbm
    num_per_class[2, :] = np.ceil(N_models/(len(exponents_dataset)-1))
    num_per_class[2, exponents_dataset == 2] = 0 # FBM can't be ballistic
    # lw
    num_per_class[3, 20:] = np.ceil((N_models/superdif)*0.8)
    # sbm
    num_per_class[4, :] = np.ceil(N_models/len(exponents_dataset))
    
    # Define return datasets
    X1 = [[],[],[]]; X2 = [[],[],[]]; X3 = [[],[],[]];
    Y1 = [[],[],[]]; Y2 = [[],[],[]]; Y3 = [[],[],[]];  
    
    # Initialize the files
    if save_dataset:
        if 1 in tasks:
            csv.writer(open(task1,'w'), delimiter=';', lineterminator='\n',)
            csv.writer(open(ref1,'w'), delimiter=';', lineterminator='\n',)
        elif 2 in tasks:
            csv.writer(open(task2,'w'), delimiter=';', lineterminator='\n',)
            csv.writer(open(ref2,'w'), delimiter=';',lineterminator='\n',)
        elif 3 in tasks:
            csv.writer(open(task3,'w'), delimiter=';', lineterminator='\n',)
            csv.writer(open(ref3,'w'), delimiter=';',lineterminator='\n',)
    
    for dim in [1,2,3]:             
        if dim not in dimensions:
            continue
    
    #%%  Generate the dataset of the given dimension
        print('Generating dataset for dimension '+str(dim)+'.')
        dataset = AD.create_dataset(T = max_T, N= num_per_class, exponents = exponents_dataset, 
                                        dimension = dim, models = np.arange(len(AD.avail_models_name)),
                                        load_trajectories = False, save_trajectories = False, N_save = 100,
                                        path = path_trajectories)            
        
    #%% Normalize trajectories

        trajs = normalize(dataset[:,2:].reshape(dataset.shape[0]*dim, max_T))
        dataset[:,2:] = trajs.reshape(dataset[:,2:].shape)

    #%% Add localization error, Gaussian noise with sigma = [0.1, 0.5, 1]
            
        loc_error_amplitude = np.random.choice(np.array(noise), size = dataset.shape[0]*dim)
        loc_error = (np.random.randn(dataset.shape[0]*dim, int((dataset.shape[1]-2)/dim)).transpose()*loc_error_amplitude).transpose()
                    
        dataset = AD.create_noisy_localization_dataset(dataset, dimension = dim, T = max_T, noise_func = loc_error)
        
    #%% Add random diffusion coefficients
        
        trajs = dataset[:,2:].reshape(dataset.shape[0]*dim, max_T)
        displacements = trajs[:,1:] - trajs[:,:-1]
        # Get new diffusion coefficients and displacements
        diffusion_coefficients = np.random.randn(trajs.shape[0])
        new_displacements = (displacements.transpose()*diffusion_coefficients).transpose()  
        # Generate new trajectories and add to dataset
        new_trajs = np.cumsum(new_displacements, axis = 1)
        new_trajs = np.concatenate((np.zeros((new_trajs.shape[0], 1)), new_trajs), axis = 1)
        dataset[:,2:] = new_trajs.reshape(dataset[:,2:].shape)
        
    
    #%% Task 1 - Anomalous exponent
        if 1 in tasks:         
            # Creating semi-balanced datasets
            N_exp = int(np.ceil(1.1*N/len(exponents_dataset)))
            for exponent in exponents_dataset:
                dataset_exp = dataset[dataset[:,1] == exponent].copy()
                np.random.shuffle(dataset_exp)
                dataset_exp = dataset_exp[:N_exp, :]
                try:
                    dataset_1 = np.concatenate((dataset_1, dataset_exp), axis = 0) 
                except:
                    dataset_1 = dataset_exp
            np.random.shuffle(dataset_1)
            dataset_1 = dataset_1[:N,:]               
            
            info_cut = []
            for traj in dataset_1[:N, :]:             
                # Cutting trajectories
                cut = np.random.randint(min_T, max_T)            
                info_cut.append(cut)            
                traj_cut = traj[2:].reshape(dim, max_T)
                traj_cut = traj_cut[:, :cut].reshape(dim*cut).tolist()               
                # Saving dataset
                X1[dim-1].append(traj_cut)
                Y1[dim-1].append(np.around(traj[1],2))
                if save_dataset:                        
                    AD.save_row(np.append(dim, traj_cut), task1)
                    AD.save_row(np.append(dim, np.around([traj[1]],2)), ref1)