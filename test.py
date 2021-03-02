from numpy import mean, corrcoef
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from generating_data import dirmake, read_ML_features, read_trajectories
from ML import split_data, load_model
from datetime import datetime
from global_params import logg, colors
from os.path import isfile
import multiprocessing as mp
from functools import partial
from TAMSD import TAMSD_estimation_traj
from math import floor
from sklearn.metrics import r2_score

def estimate(test_data, test_label, part, Model, Model_test, model, learning_number = 100000, part_test = None):
    if part_test == None:
        part_test = part
        path_estimations = f'data/part{part}/estimations/model{Model}/test{Model_test}/{model}.csv'
        dirmake(f'data/part{part}/estimations/model{Model}/test{Model_test}/')
    elif part_test >= 3:
        path_estimations = f'data/part1-6/estimations/part{part_test}/{model}.csv'
        dirmake(f'data/part1-6/estimations/part{part_test}/')
    elif part_test == 1:
        if part == 3:
            path_estimations = f'data/part3-1/estimations/part{part_test}/{model}.csv'
            dirmake(f'data/part3-1/estimations/part{part_test}/')
        elif part in [1,7,8,9,10]:
            path_estimations = f'data/part7-10/estimations/part{part}/{model}.csv'
            dirmake(f'data/part7-10/estimations/part{part}/')
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
            print(len(test_label))
            results = pd.DataFrame({'expo': test_label, 'expo_est': predicted_labels})
            print(' --- ZAKOŃCZONO')
            print(f'Zapisywanie wstymacji wyników do pliku - model {model}...')
            results.to_csv(path_estimations)
            print(' --- ZAKOŃCZONO')
            print(64 * '-')
    results = results.dropna()
    return list(results['expo_est']), list(results['expo'])

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
        plt.title(' ', loc='left')
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\hat\alpha$')
        plt.xlim(0,2)
        plt.ylim(0,2)
        plt.savefig(path+model_name+'cont.pdf', transparent = True, bbox_inches = 'tight', dpi = 300)
    return strength, errors


def test_models(part, Model, num_of_learning = 100000, tests = None, mode = 'full'):
    if tests == None:
        tests = Model
    if part >= 1:
        path_results = f'data/part{part}/results/model{Model}/test{tests}/'
        path_estimate = f'data/part{part}/estimations/model{Model}/test{tests}/'
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
                estimated_label, test_labels = estimate(trajectories, test_label, part, Model, tests, 'TAMSD')
            else:
                estimated_label, test_labels = estimate(test_data, test_label, part, Model, tests, model)
            # # # określanie mocy
            table, er = test_model(estimated_label, test_labels, model.replace('_', ' '), path_results, mode)
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
            labels = ['TAMSD', 'LR', 'DT', 'RF', 'GB']
            box = plt.boxplot(labels = labels, x = err.values(), patch_artist=True, notch = True, vert = True)
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
     
def test_models2(part, part_test, num_of_learning = 100000, mode = 'full'):
    ''' Do testowania modelu z częsi 1 do danych z części 3-6 i do testowania modelu\
        z części 3 na danych z części 1 i do testowania modeli części 7-10 na części 1'''
    Model = 'A'; tests = 'A'
    if part == 1:
        if part_test in [3,4,5,6]:
            path_results = f'data/part1-6/results/part'+str(part_test)+'/'
            path_estimate = f'data/part1-6/estimations/part'+str(part_test)+'/'
    elif part == 3:
        if part_test == 1:
            path_results = f'data/part3-1/results/part'+str(part_test)+'/'
            path_estimate = f'data/part3-1/estimations/part'+str(part_test)+'/'
    if part in [1,7,8,9,10]:
        if part_test == 1:
            path_results = f'data/part7-10/results/part'+str(part)+'/'
            path_estimate = f'data/part7-10/estimations/part'+str(part)+'/'
    dataset = read_ML_features(part_test, tests)
    if not 'no_gen' in mode:
        trajectories = read_trajectories(part_test, tests, 'end', num_of_learning) # to test
    else:
        trajectories = 'trajectories'
    _, _, test_data, test_label = split_data(dataset, num_of_learning)
    
    test_label = list(test_label)
    
    dirmake(path_results)
    dirmake(path_estimate)
    
    models = ['TAMSD', 'linear_regression', 'decision_tree', 'random_forest', 'gradient_boosting']
    
    ttable = pd.DataFrame()
    err = {}
    
    plt.figure(figsize = (2,2))
    
    for model in models:
        # # # estymowanie parametrów
        if model == 'TAMSD':
            estimated_label, test_labels = estimate(trajectories, test_label, part, Model, tests, 'TAMSD', 100000, part_test)
        else:
            estimated_label, test_labels = estimate(test_data, test_label, part, Model, tests, model, 100000, part_test)
        # # # określanie mocy
        table, er = test_model(estimated_label, test_labels, model.replace('_', ' '), path_results, mode)
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

#
# ### part 1-2 ### #
#

# part = 2
# for Model in ['A']:
#     for Model_test in ['A', 'B']:
#         print(f'Praca nad {Model} - {Model_test}')
#         test_models(part, Model, 100000, Model_test)

#
# ### part 3-6 ### #
#

# Model = 'A'
# Model_test = 'A'
# for part in [3,4,5,6]:
#     print(f'Praca nad {part}')
#     test_models(part, Model, 100000, Model_test)

# models = ['linear_regression', 'decision_tree', 'random_forest', 'gradient_boosting']
# parts = [3,4,5,6]
# rozdzial = 3
# def plot_noisy():
#     normal_models = {}
#     for model in models:
#         table_path = 'data/part1/results/modelA/testA/table.csv'
#         table = pd.read_csv(table_path, index_col = 'Unnamed: 0')
#         line = table.loc[model.replace('_',' ')]
#         normal_models[model] = {'R2': line['R^2'], 'MAE': line['MAE'], 'MSE': line['MSE']}
#     path_results = 'data/part3-6/'
#     dirmake(path_results)
#     R2s = {}
#     MAEs = {}
#     MSEs = {}
#     for model in models:
#         R2 = []; MAE = []; MSE = []
#         for part in parts:
#             path_table = 'data/part'+str(part)+'/results/modelA/testA/table.csv'
#             table = pd.read_csv(path_table, index_col = 'Unnamed: 0')
#             R2.append(table.loc[model.replace('_',' ')]['R^2'])
#             MAE.append(table.loc[model.replace('_',' ')]['MAE'])
#             MSE.append(table.loc[model.replace('_',' ')]['MSE'])
#         R2s[model] = R2
#         MAEs[model] = MAE
#         MSEs[model] = MSE
#         for stat in ['R2', 'MAE', 'MSE']:
#             statistica = eval(stat)
#             plt.clf()
#             plt.figure(figsize = (4,4))
#             plt.plot([0,0.1,0.3,1], statistica, label = stat)
#             plt.title(model, loc = 'left')
#             plt.xticks([0,0.1,0.3,1])
#             plt.yticks(statistica)
#             plt.xlim(0,1)
#             dirmake(path_results+'/'+stat+'/')
#             plt.savefig(path_results+stat+'/'+model+'.pdf')
#     for stat in ['R2', 'MAE', 'MSE']:
#         statistica = eval(stat+'s')
#         path_result = path_results + stat+'.pdf'
#         plt.close('all')
#         plt.figure(figsize = (3,3))
#         n = 0
#         for key, value in statistica.items():
#             plt.plot([0,0.1,0.3,1], value, label = key.replace('_', ' '), c=colors[n+1], marker = 'x')
#             plt.plot([0,1], [normal_models[key][stat]] * 2, '--',
#                      label = key.replace('_', ' ')+' w r.'+str(rozdzial), c = colors[n+1])
#             n += 1
#         plt.xticks([0,0.1,0.3,1], labels=['0','0.1','0.3','1'])
#         if stat == 'MSE':
#             plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
#         plt.title(stat, loc = 'left')
#         plt.ylim(0.3,0.9)
#         plt.savefig(path_result, transparent = True, bbox_inches = 'tight', dpi = 300)
    
# plot_noisy()

#
# ### part 1-6 ### #
#

# Model = 'A'
# Model_test = 'A'
# for part_test in [3,4,5,6]:
#     print(f'Praca nad {part_test}')
#     test_models2(1, part_test, 100000)
    
# models = ['linear_regression', 'decision_tree', 'random_forest', 'gradient_boosting']
# labels = ['regresja liniowa', 'drzewo decyzyjne', 'las losowy', 'wzmocnienie gradientowe']
# parts = [3,4,5,6]
# rozdzial = 3
# def plot_noisy2():
#     normal_models = {}
#     for model in models:
#         table_path = 'data/part1/results/modelA/testA/table.csv'
#         table = pd.read_csv(table_path, index_col = 'Unnamed: 0')
#         line = table.loc[model.replace('_',' ')]
#         normal_models[model] = {'R2': line['R^2'], 'MAE': line['MAE'], 'MSE': line['MSE']}
#     path_results = f'data/part1-6/'
#     dirmake(path_results)
#     R2s = {}
#     MAEs = {}
#     MSEs = {}
#     for model in models:
#         R2 = []; MAE = []; MSE = []
#         for part_test in parts:
#             path_table = 'data/part1-6/results/part'+str(part_test)+'/table.csv'
#             table = pd.read_csv(path_table, index_col = 'Unnamed: 0')
#             R2.append(table.loc[model.replace('_',' ')]['R^2'])
#             MAE.append(table.loc[model.replace('_',' ')]['MAE'])
#             MSE.append(table.loc[model.replace('_',' ')]['MSE'])
#         R2s[model] = R2
#         MAEs[model] = MAE
#         MSEs[model] = MSE
#         for stat in ['R2', 'MAE', 'MSE']:
#             statistica = eval(stat)
#             plt.clf()
#             plt.figure(figsize = (4,4))
#             plt.plot([0,0.1,0.3,1], statistica, label = stat)
#             plt.title(model, loc = 'left')
#             plt.xticks([0,0.1,0.3,1])
#             plt.yticks(statistica)
#             plt.xlim(0,1)
#             dirmake(path_results+'/'+stat+'/')
#             plt.savefig(path_results+stat+'/'+model+'.pdf')
#     for stat in ['R2', 'MAE', 'MSE']:
#         statistica = eval(stat+'s')
#         path_result = path_results + stat+'.pdf'
#         plt.close('all')
#         plt.figure(figsize = (3,3))
#         n = 0
#         for key, value in statistica.items():
#             plt.plot([0,0.1,0.3,1], value, label = labels[n], c=colors[n+1], marker = 'x')
#             plt.plot([0,1], [normal_models[key][stat]] * 2, '--',
#                      label = labels[n]+r' $($m$/$m$)$', c = colors[n+1])
#             n += 1
#         plt.xticks([0,0.1,0.3,1], labels=['0','0.1','0.3','1'])
#         if stat == 'MSE':
#             plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
#         plt.title(stat, loc = 'left')
#         plt.ylim(0.3,0.9)
#         plt.savefig(path_result, transparent = True, bbox_inches = 'tight', dpi = 300)
    
# plot_noisy2()

#
# ### part 3-1 ### #
#

# Model = 'A'
# Model_test = 'A'
# part = 3
# part_test = 1
# test_models2(part, part_test, 100000)

#
# ### part 7-10 ### #
#

# def plot_time1(path):
#     t = [10,100,1000,10000,100000]
#     LR_learn = [0.006295,
#                 0.0003375,
#                 0.002698,
#                 0.00268,
#                 0.02188]
#     DT_hiper = [2.243328,
#                 0.609186,
#                 1.208129,
#                 5.116622,
#                 7.777683]
#     DT_learn = [0.005475,
#                 0.002901,
#                 0.021804,
#                 0.151335,
#                 1.441862]
#     RF_hiper = [39.0821,
#                 40.7799,
#                 61.5614,
#                 410.0553,
#                 513.717841]
#     RF_learn = [0.3253,
#                 0.2588,
#                 1.6796,
#                 12.2012,
#                 272.040734]
#     GB_hiper = [11.010089,
#                 21.120445,
#                 113.096703,
#                 1150.180638,
#                 1342.963843]
#     GB_learn = [0.0599,
#                 0.1231,
#                 1.1579,
#                 5.4285,
#                 77.559463]
#     _data = [[], DT_hiper, RF_hiper, GB_hiper, LR_learn, DT_learn, RF_learn, GB_learn]
#     _data_labels = [' ',
#                     'DT — szukanie hiperparametrów',
#                     'RF — szukanie hiperparametrów',
#                     'GB — szukanie hiperparametrów',
#                     'LR — nauczanie',
#                     'DT — nauczanie',
#                     'RF — nauczanie',
#                     'GB — nauczanie']
#     plt.cla()
#     plt.figure(figsize = (5,5))
#     for i in range(len(_data)):
#         style = {True: '' ,False: '--'}[i >= 4]
#         data = _data[i]
#         data_labels = _data_labels[i]
#         if i > 0:
#             plt.loglog(t, data, style, label = data_labels, c = colors[1+i%4])
#         else:
#             plt.loglog([], data, style, label = data_labels, c = 'white')
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
#     plt.xlabel('ilość danych treningowych')
#     plt.ylabel('czas [s]')
#     plt.xticks = t
#     plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol = 2)
#     plt.savefig(path, transparent = True, bbox_inches = 'tight', dpi = 300)
# dirmake('data/part7-10/')
# plot_time1('data/part7-10/timing.pdf')

# Model = 'A'
# Model_test = 'A'
# part_test = 1
# for part in [7,8,9,10,1]:
#     print(f'Praca nad {part}')
#     test_models2(part, part_test, 100000, 'table')

models = ['linear_regression', 'decision_tree', 'random_forest', 'gradient_boosting']
test_part = 1
parts = [7,8,9,10,1]
rozdzial = 3
def plot_noisy3():
    normal_models = {}
    path_results = 'data/part7-10/'
    dirmake(path_results)
    R2s = {}
    MAEs = {}
    MSEs = {}
    for model in models:
        R2 = []; MAE = []; MSE = []
        for part in parts:
            path_table = 'data/part7-10/results/part'+str(part)+'/table.csv'
            table = pd.read_csv(path_table, index_col = 'Unnamed: 0')
            R2.append(table.loc[model.replace('_',' ')]['R^2'])
            MAE.append(table.loc[model.replace('_',' ')]['MAE'])
            MSE.append(table.loc[model.replace('_',' ')]['MSE'])
        R2s[model] = R2
        MAEs[model] = MAE
        MSEs[model] = MSE
        for stat in ['R2', 'MAE', 'MSE']:
            statistica = eval(stat)
            plt.clf()
            plt.figure(figsize = (4,4))
            plt.semilogx([10,100,1000,10000,100000], statistica, label = stat)
            plt.title(model, loc = 'left')
            plt.xticks([10,100,1000,10000,100000])
            plt.yticks(statistica)
            dirmake(path_results+'/'+stat+'/')
            plt.savefig(path_results+stat+'/'+model+'.pdf')
    for stat in ['R2', 'MAE', 'MSE']:
        statistica = eval(stat+'s')
        path_result = path_results + stat+'.pdf'
        plt.close('all')
        plt.figure(figsize = (3,3))
        n = 0
        for key, value in statistica.items():
            if stat == 'R2':
                plt.semilogx([10,100,1000,10000,100000], value, label = key.replace('_', ' '), c=colors[n+1], marker = 'x')
            else:
                plt.loglog([10,100,1000,10000,100000], value, label = key.replace('_', ' '), c=colors[n], marker = 'x')
            n += 1
        plt.xticks([10,100,1000,10000,100000])
        if stat == 'MSE':
            plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        plt.title(stat, loc = 'left')
        plt.savefig(path_result, transparent = True, bbox_inches = 'tight', dpi = 300)
plot_noisy3()