from global_params import N_long, logg
from generating_data import generate_trajectories, read_trajectories, dirmake, read_TAMSD, read_real_expo, read_ML_features
from plotting import plot_traj, plot_TAMSD
from TAMSD import TAMSD_estimation
from get_features import get_features
from ML import linear_regression, decision_tree, random_forest, gradient_boosting
from os import remove
from datetime import datetime
from part0 import example_trajs, example_TAMSD

remove('log.txt')

with open('log.txt', 'w') as f:
    f.write(f'[{datetime.now()}] START programu...\n')

colors = ['darkorange', 'darkgreen', 'royalblue', 'pink', 'red', 'magenta'] # kolory do rysowania plotów
part = int(input("Jaką część projektu chcesz uruchomić: "))
logg(f'Starting part {part}.')
if part == 0:
    Q_trajectories = input("Czy chcesz wygenerować przykłady trajektorii? (Y/n): ")
    Q_TAMSD = input("Czy chcesz wygenerować przykładowe i prerfekcyjne TAMSD? (Y/n): ")
    logg(f'Wybrane decyzje: {Q_trajectories}.')
    print(64 * '-')
    if Q_trajectories == 'Y':
        example_trajs()
        print(64 * '-')
    if Q_TAMSD == 'Y':
        example_TAMSD()
    
 if part in [1,2,3,4,5,6,7,8,9,10]:
    Model = input('Jaki model chcesz uruchomić? (A/B/C): ')
    
    logg(f'Wybrany model: {Model}.')
    
    Q_generate = input("Czy chcesz wygenerować dane? (Y/n): ")
    Q_generate_plot = int(input("Ile trajektorii zapisać w postaci graficznej: "))
    Q_TAMSD = input("Czy chcesz przeliczyć TAMSD? (Y/n): ")
    Q_TAMSD_plot = int(input("Ile obliczonych TAMSD chcesz zapisać w postaci graficznej: "))
    Q_ML_features = input("Czy chcesz wyciągnąć paramtry trajektorii do ML? (Y/n): ")
    Q_ML_linreg = input("Czy chcesz użyć wielowymiarowej regresji liniowej? (Y/n): ")
    Q_ML_dectree = input("Czy chcesz użyć decision tree? (Y/n): ")
    Q_ML_randomforest = input("Czy chcesz użyć random forest? (Y/n): ")
    Q_ML_gradientboosting = input("Czy chcesz użyć gradient boosting? (Y/n): ")

    logg(f'Wybrane decyzje: {Q_generate}, {Q_generate_plot}, {Q_TAMSD}, {Q_TAMSD_plot}'+
        f', {Q_ML_features}, {Q_ML_linreg}, {Q_ML_dectree}, {Q_ML_randomforest}, {Q_ML_gradientboosting}.')
    print(64 * '-')
    
    N = N_long
    path = f'data/part{part}/model{Model}/'
    dirmake(path)
    traject_loaded = False
    expo_loaded = False
    features_loaded = False
    
    if Q_generate == 'Y': 
        # generowanie trajektorii
        generate_trajectories(N, part, Model)
        print(64 * '-')
        
    if Q_generate_plot > 0:
        # rysowanie trajektorii
        if not traject_loaded:
            trajectories = read_trajectories(part, Model)
            traject_loaded = True
        print(f'Tworzenie {Q_generate_plot} wykresów...')
        dirmake(path+'generated/traj_plot/')
        n = 0
        for traj in trajectories[:Q_generate_plot]:
            plot_traj(*traj, path+'generated/traj_plot/plot_1_'+str(n)+'.pdf', {'color': colors[n]}, {'figsize': (2,2)})
            n += 1
        print(64 * '-')
        
    if Q_TAMSD == 'Y':
        # wykonywanie TAMSD
        if not traject_loaded:
            trajectories = read_trajectories(part, Model)
            traject_loaded = True
        if not expo_loaded:
            exps = read_real_expo(part, Model)
        TAMSD_estimation(trajectories, exps, part, Model)
        print(64 * '-')
        
    if Q_TAMSD_plot > 0:
        # rysowanie trajektorii
        tamsd_data = read_TAMSD(part, Model)
        print(f'Tworzenie {Q_TAMSD_plot} wykresów...')
        dirmake(path+'TAMSD/plot/')
        n = 0
        for i in range(Q_TAMSD_plot):
            plot_TAMSD(list(tamsd_data.loc[i]), path+'TAMSD/plot/tamsd'+str(i)+'.pdf', {'figsize': (4,3)}, ['a','b','c','d','e','f'][i])
            n += 1
        print(' --- ZAKOŃCZONO')
        print(64 * '-')
        
    if Q_ML_features == 'Y':
        # wyciąganie danych
        if not traject_loaded:
            trajectories = read_trajectories(part, Model)
            traject_loaded = True
        if not expo_loaded:
            exps = read_real_expo(part, Model)
        get_features(trajectories, exps, part, Model)
        print(64 * '-')
        
    if Q_ML_linreg == 'Y':
        if not features_loaded:
            features = read_ML_features(part, Model)
        linear_regression(features, part, Model)
        print(64 * '-')
        
    if Q_ML_dectree == 'Y':
        if not features_loaded:
            features = read_ML_features(part, Model)
        decision_tree(features, part, Model)
        print(64 * '-')
        
    if Q_ML_randomforest == 'Y':
        if not features_loaded:
            features = read_ML_features(part, Model)
        random_forest(features, part, Model)
        print(64 * '-')
        
    if Q_ML_gradientboosting == 'Y':
        if not features_loaded:
            features = read_ML_features(part, Model)
        gradient_boosting(features, part, Model)
        print(64 * '-')
