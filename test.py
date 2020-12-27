from numpy import mean, corrcoef
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from generating_data import dirmake, read_ML_features, read_trajectories
from ML import split_data, load_model
from datetime import datetime
from global_params import logg
from os.path import isfile
import multiprocessing as mp
from functools import partial
from TAMSD import TAMSD_estimation_traj

def estimate(test_data, test_label, part, Model, Model_test, model, learning_number = 100000):
    path_estimations = f'data/part{part}/estimations/model{Model}/test{Model_test}/{model}.csv'
    if isfile(path_estimations):
        results = pd.read_csv(path_estimations, index_col='Unnamed: 0')
    else:
        if model == 'TAMSD':
            print('Obliczanie estymacji TAMSDS...')
            logg('TAMSD - estymacja - start')
            start = datetime.now()
            traj_num = len(test_label)
            give = []
            for i in range(traj_num):
                give.append([test_label[i], test_data[i]])
            
            with mp.Pool(3) as pool:
                temp = partial(TAMSD_estimation_traj, part, traj_num)
                result = pool.map(temp, give)
                pool.close()
                pool.join()
            stop = datetime.now()
            logg(f'TAMSD - estymacja - koniec {stop - start}')
            print(' --- ZAKOŃCZONO')
            
            print('Translacja wyników TAMSD...')
            results = pd.DataFrame(columns = ['D', 'expo', 'expo_est', 'tamsds'],
                            index = range(traj_num))
            liczydlo = 0
            for i in result:
                results.loc[liczydlo] = i
                liczydlo += 1
            print(' --- ZAKOŃCZONO')
            results.to_csv(path_estimations)
                
        else:
            path_model = f'data/part{part}/model{Model}/ML/{model}/model.pk1'
            print('Ładowanie modelu ...')
            model = load_model(path_model)
            print(' --- ZAKOŃCZONO')
            print(f'Testowanie modelu {model}...')
            logg(f'ML - {model} - przewidywanie - start')
            start = datetime.now()
            predicted_labels = model.predict(test_data)
            stop = datetime.now()
            logg(f'ML - {model} - przewidywanie - koniec {stop - start}')
            print(' --- ZAKOŃCZONO')
            print('Translacja przewidywań...')
            results = pd.DataFrame({'expo': test_label, 'expo_est': predicted_labels})
            print(' --- ZAKOŃCZONO')
            print(f'Zapisywanie wstymacji wyników do pliku - model {model}...')
            results.to_csv(path_estimations)
            print(' --- ZAKOŃCZONO')
            print(64 * '-')
    return list(results['expo_est'])

def test_model(predicted_values, real_values, model_name, path, mode):
    N = len(predicted_values)
    ## R^2
    R2 = (corrcoef(real_values, predicted_values)[0,1]) ** 2
    MAE = 0
    MSE = 0
    ## errors
    errors = []
    for i in range(N):
        MAE += abs(real_values[i] - predicted_values[i]) / N
        MSE += abs(real_values[i] - predicted_values[i]) ** 2 / N
        ## errors
        errors.append(real_values[i] - predicted_values[i])
    strength = pd.DataFrame({'R^2': R2,
                             'MAE': MSE,
                             'MSE': MAE
                            },
                            index = [model_name])

    if 'full' in mode or 'plot' in mode:
        plt.clf()
        sns.displot({'r': real_values, 'e': predicted_values}, x = 'r', y = 'e')
        plt.plot([0,2], [0,2], color = 'black')
        plt.title(model_name, loc='left')
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\hat\alpha$')
        plt.xlim(0,2)
        plt.ylim(0,2)
        plt.savefig(path+model_name+'.pdf', transparent = True, bbox_inches = 'tight', dpi = 300)
        # contours
        plt.clf()
        sns.displot({'r': real_values, 'e': predicted_values}, x = 'r', y = 'e', kind="kde", cmap=sns.light_palette("red", as_cmap=True))
        plt.plot([0,2], [0,2], color = 'black')
        plt.title(model_name, loc='left')
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\hat\alpha$')
        plt.xlim(0,2)
        plt.ylim(0,2)
        plt.savefig(path+model_name+'cont.pdf', transparent = True, bbox_inches = 'tight', dpi = 300)
    return strength, errors


def test_models(part, Model, num_of_learning = 100000, tests = None, mode = 'full'):
    if tests == None:
        tests = Model
    if part == 1 or part == 2:
        path_results = f'data/part{part}/results/model{Model}/test{tests}/'
        path_estimate = f'data/part{part}/estimations/model{Model}/test{Model_test}/'
        dataset = read_ML_features(part, tests)
        if not 'no_gen' in mode:
            trajectories = read_trajectories(part, tests, 'end', num_of_learning)
        else:
            trajectories = 'trajectories'
        _, _, test_data, test_label = split_data(dataset, num_of_learning)
        
        test_label = list(test_label)
        
        dirmake(path_results)
        dirmake(path_estimate)
        
        models = ['TAMSD', 'linear_regression', 'decision_tree', 'random_forest', 'gradient_boosting']
        
        ttable = pd.DataFrame(columns = ['R^2', 'eps = 0.05', 'eps = 0.025', 'max_error'])
        err = {}
        
        plt.figure(figsize = (2,2))
        
        for model in models:
            # # # estymowanie parametrów
            if model == 'TAMSD':
                estimated_label = estimate(trajectories, test_label, part, Model, tests, 'TAMSD')
            else:
                estimated_label = estimate(test_data, test_label, part, Model, tests, model)
            # # # określanie mocy
            table, er = test_model(estimated_label, test_label, model.replace('_', ' '), path_results, mode)
            if 'full' in mode or 'table' in mode:
                ttable = pd.concat([ttable, table])
            err[model] = er
            print(f'Zrobione {Model} - {tests} - {model}')
            # # # rysowanki
            
        if 'full' in mode or 'table' in mode:
            print(f'Zapisywanie tabeli do pliku {path_results+"table.csv"}')
            ttable.to_csv(path_results+'table.csv')
            
        print(f'Zrobione tabele dla {Model}-{tests}')
            
        if 'full' in mode or 'plot' in mode:
            # box plot
            plt.clf()
            plt.figure(figsize = (5,5))
            box = plt.boxplot(labels = err.keys(), x = err.values(), patch_artist=True, notch = True, vert = True)
            colors = ['lightblue', 'orange', 'lightgreen', 'pink', 'violet']
            for plott, color in zip(box['boxes'], colors):
                print(plott)
                plott.set_facecolor(color)
            plt.savefig(path_results+'boxplots.pdf', transparent = True, bbox_inches = 'tight', dpi = 300)
            
            # displots
            plt.clf()
            sns.displot(err, common_norm=False, stat="density")
            plt.plot([0,0], [0,1.3], color = 'black')
            plt.savefig(path_results+'displot1.pdf', transparent = True, bbox_inches = 'tight', dpi = 300)

            plt.clf()
            sns.displot(err, kind = "kde", common_norm=False, bw_adjust = 0.7, fill=True)
            plt.plot([0,0], [0,1.3], color = 'black')
            plt.savefig(path_results+'displot2.pdf', transparent = True, bbox_inches = 'tight', dpi = 300)
            
            plt.clf()
            sns.displot(err, kind = "kde", common_norm=False, bw_adjust=2)
            plt.plot([0,0], [0,1.3], color = 'black')
            plt.savefig(path_results+'displot3.pdf', transparent = True, bbox_inches = 'tight', dpi = 300)

            plt.clf()
            sns.displot(err, kind = "ecdf")
            plt.plot([0,0], [0,1.3], color = 'black')
            plt.savefig(path_results+'displot4.pdf', transparent = True, bbox_inches = 'tight', dpi = 300)
            
        print(f'Zrobione wykresy dla {Model}-{tests}')

part = 2
for Model in ['A', 'B', 'C']:
    for Model_test in ['A', 'B', 'C']:
        print(f'Praca nad {Model} - {Model_test}')
        test_models(part, Model, 100000, Model_test)