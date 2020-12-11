from global_params import T_long, N_long
from generating_data import generate_trajectories, read_trajectories, dirmake, read_TAMSD, read_real_expo, read_ML_features
from plotting import plot_traj, plot_TAMSD
from TAMSD import TAMSD_estimation
from get_features import get_features
from ML import linear_regression, decision_tree, random_forest

colors = ['darkorange', 'darkgreen', 'royalblue'] # kolory do rysowania plotów
part = int(input("Jaką część projektu chcesz uruchomić: "))
if part == 1:
    Q_generate = input("Czy chcesz wygenerować dane? (Y/n): ")
    Q_generate_plot = int(input("Ile trajektorii zapisać w postaci graficznej: "))
    Q_TAMSD = input("Czy chcesz przeliczyć TAMSD? (Y/n): ")
    Q_TAMSD_plot = int(input("Ile obliczonych TAMSD chcesz zapisać w postaci graficznej: "))
    Q_ML_features = input("Czy chcesz wyciągnąć paramtry trajektorii do ML? (Y/n): ")
    Q_ML_linreg = input("Czy chcesz użyć wielowymiarowej regresji liniowej? (Y/n): ")
    Q_ML_dectree = input("Czy chcesz użyć decision tree? (Y/n): ")
    Q_ML_randomforest = input("Czy chcesz użyć random forest? (Y/n): ")
    print(64 * '-')
    T = T_long
    N = N_long
    path = 'data/part1/'
    traject_loaded = False
    expo_loaded = False
    features_loaded = False
    if Q_generate == 'Y': 
        # generowanie trajektorii
        generate_trajectories(N, T, part)
        print(64 * '-')
    if Q_generate_plot > 0:
        # rysowanie trajektorii
        if not traject_loaded:
            trajectories = read_trajectories(part)
            traject_loaded = True
        print(f'Tworzenie {Q_generate_plot} wykresów...')
        dirmake(path+'generated/traj_plot/')
        n = 0
        for traj in trajectories[:Q_generate_plot]:
            plot_traj(*traj, path+'generated/traj_plot/plot_1_'+str(n)+'.png', {'color': colors[n]}, {'figsize': (2,2)})
            n += 1
        print(64 * '-')
    if Q_TAMSD == 'Y':
        # wykonywanie TAMSD
        if not traject_loaded:
            trajectories = read_trajectories(part)
            traject_loaded = True
        if not expo_loaded:
            exps = read_real_expo(part)
        TAMSD_estimation(trajectories, exps, part)
        print(64 * '-')
    if Q_TAMSD_plot > 0:
        # rysowanie trajektorii
        tamsd_data = read_TAMSD(part)
        print(f'Tworzenie {Q_TAMSD_plot} wykresów...')
        dirmake(path+'TAMSD/plot/')
        n = 0
        for i in range(Q_TAMSD_plot):
            plot_TAMSD(list(tamsd_data.loc[i]), path+'TAMSD/plot/tamsd'+str(i)+'.png', {'figsize': (4,3)}, ['a','b','c','d','e','f'][i])
            n += 1
        print(' --- ZAKOŃCZONO')
        print(64 * '-')
    if Q_ML_features == 'Y':
        # wyciąganie danych
        if not traject_loaded:
            trajectories = read_trajectories(part)
            traject_loaded = True
        if not expo_loaded:
            exps = read_real_expo(part)
        get_features(trajectories, exps, part)
        print(64 * '-')
    if Q_ML_linreg == 'Y':
        if not features_loaded:
            features = read_ML_features(part)
        linear_regression(features, part)
    if Q_ML_dectree == 'Y':
        if not features_loaded:
            features = read_ML_features(part)
        decision_tree(features, part)
    if Q_ML_randomforest == 'Y':
        if not features_loaded:
            features = read_ML_features(part)
        random_forest(features, part)
