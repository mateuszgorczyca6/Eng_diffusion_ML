# Remember to use other env
# # # Linux:
# # Installing venv
# pip3 install venv
# # Creating enviroment
# python3 -m venv <env_name>
# # Activating enviroment
# source ~/<env_name>/bin/activate
#
# My env is 'env_for_keras_gpu'
#
# # # Imports # # #
#
from global_params import number_to_learn
from sklearn.linear_model import LinearRegression
from pickle import dump, load
from generating_data import dirmake
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from global_params import logg as log
from datetime import datetime
from math import floor

def split_data(dataset, learning_part):
    train_dataset = dataset[:learning_part]
    test_dataset = dataset.drop(train_dataset.index)
    train_labels = train_dataset.pop('alpha')
    test_labels = test_dataset.pop('alpha')
    return train_dataset, train_labels, test_dataset, test_labels

def save_model(model, path):
    dirmake('/'.join(path.split('/')[:-1]))
    with open(path, 'wb') as f:
        dump(model, f)

def load_model(path):
    with open(path, 'rb') as f:
        model = load(f)
    return model

def linear_regression(features, part, Model):
    train_data, train_labels, test_data, test_label = split_data(features, number_to_learn, part)
    print('Wyznaczanie modelu wilowymiarowej regresji liniowej...')
    log(f'ML - regresja liniowa - nauczanie - start')
    start = datetime.now()
    model = LinearRegression(normalize = True, n_jobs=-1)
    model = model.fit(train_data, train_labels)
    stop = datetime.now()
    log(f'ML - regresja liniowa - nauczanie - koniec {stop - start}')
    print(' --- ZAKOŃCZONO')
    path = f'data/part1/model{Model}/ML/linear_regression/model.pk1'
    print('Zapisywanie modelu do pliku {}'.format(path))
    save_model(model, path)
    print(' --- ZAKOŃCZONO')
    print('Testowanie modelu wielowymiarowej regresji liniowej...')
    log(f'ML - regresja liniowa - przewidywanie - start')
    start = datetime.now()
    predicted_labels = model.predict(test_data)
    stop = datetime.now()
    log(f'ML - regresja liniowa - przewidywanie - koniec {stop - start}')
    print(' --- ZAKOŃCZONO')
    print('Translacja przewidywań...')
    results = pd.DataFrame({'expo': test_label, 'expo_est': predicted_labels})
    print(' --- ZAKOŃCZONO')
    print('Translacja wyników do pliku...')
    results.to_csv(f'data/part1/model{Model}/ML/linear_regression/estimated.csv')
    print(' --- ZAKOŃCZONO')

def decision_tree(features, part, Model):
    train_data, train_labels, test_data, test_label = split_data(features, number_to_learn, part)
    hiperparam_data = train_data[:floor(number_to_learn/10)]
    hiperparam_labels = train_labels[:floor(number_to_learn/10)]
    print('Wyznaczanie drzewa decyzyjnego...')
    max_depth = list(range(2,20,1))
    min_samples_split = list(range(1,11))
    min_samples_leaf = list(range(1,6))
    max_features = ['auto', 'sqrt']
    random_grid = {'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'max_features': max_features}
    log(f'ML - drzewo decyzyjne - szukanie superparametrów - start')
    start = datetime.now()
    model = DecisionTreeRegressor()
    model = RandomizedSearchCV(estimator = model,
                               param_distributions = random_grid,
                               n_iter = 30,
                               cv = 3,
                               verbose=2,
                               random_state = 42,
                               n_jobs = 3,
                               return_train_score=True,
                               refit = True)
    model = model.fit(hiperparam_data, hiperparam_labels)
    stop = datetime.now()
    log(f'ML - drzewo decyzyjne - szukanie superparametrów - koniec {stop - start}')
    model_params = pd.DataFrame(model.best_params_, index = ['decision tree'])
    dirmake(f'data/part1/model{Model}/ML/decision_tree')
    model_params.to_csv(f'data/part1/model{Model}/ML/decision_tree/model_params.csv')
    model = DecisionTreeRegressor(**model.best_params_)
    log(f'ML - drzewo decyzyjne - nauczanie - start')
    start = datetime.now()
    model.fit(train_data, train_labels)
    stop = datetime.now()
    log(f'ML - drzewo decyzyjne - nauczanie - koniec {stop - start}')
    print(' --- ZAKOŃCZONO')
    path = f'data/part1/model{Model}/ML/decision_tree/model.pk1'
    print('Zapisywanie modelu do pliku {}'.format(path))
    save_model(model, path)
    plt.cla()
    plt.figure(figsize=(10,6.5))
    plot_tree(model, max_depth = 3, feature_names = list(test_data), fontsize=10, filled=True)
    path = f'data/part1/model{Model}/ML/decision_tree/tree.pdf'
    plt.savefig(path, transparent = True, bbox_inches = 'tight')
    plt.cla()
    plt.figure(figsize=(15,15))
    plot_tree(model, feature_names = list(test_data), filled=True)
    path = f'data/part1/model{Model}/ML/decision_tree/full_tree.pdf'
    plt.savefig(path, transparent = True, bbox_inches = 'tight')
    print(' --- ZAKOŃCZONO')
    print('Testowanie modelu drzewa decyzyjnego...')
    log(f'ML - drzewo decyzyjne - przewidywanie - start')
    start = datetime.now()
    predicted_labels = model.predict(test_data)
    stop = datetime.now()
    log(f'ML - drzewo decyzyjne - przewidywanie - koniec {stop - start}')
    print(' --- ZAKOŃCZONO')
    print('Translacja przewidywań...')
    results = pd.DataFrame({'expo': test_label, 'expo_est': predicted_labels})
    print(' --- ZAKOŃCZONO')
    print('Translacja wyników do pliku...')
    results.to_csv(f'data/part1/model{Model}/ML/decision_tree/estimated.csv')
    print(' --- ZAKOŃCZONO')

def random_forest(features, part, Model):
    train_data, train_labels, test_data, test_label = split_data(features, number_to_learn, part)
    hiperparam_data = train_data[:floor(number_to_learn/10)]
    hiperparam_labels = train_labels[:floor(number_to_learn/10)]
    print('Wyznaczanie modelu random forest...')
    # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    n_estimators = list(range(100,1001,100))
    max_depth = list(range(2,20,1))
    min_samples_split = list(range(2,11))
    min_samples_leaf = list(range(1,6))
    max_features = ['auto', 'sqrt']
    max_samples = [0.1*i for i in range(1, 10)]
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'max_features': max_features,
                   'max_samples': max_samples}
    model = RandomForestRegressor(bootstrap = True, max_samples=0.5)
    log(f'ML - las losowy - szukanie superparametrów - start')
    start = datetime.now()
    model = RandomizedSearchCV(estimator = model,
                               param_distributions = random_grid,
                               n_iter = 30,
                               cv = 3,
                               verbose=2,
                               random_state = 42,
                               n_jobs = 3,
                               return_train_score=True,
                               refit = True)
    model.fit(hiperparam_data, hiperparam_labels)
    stop = datetime.now()
    log(f'ML - las losowy - szukanie superparametrów - koniec {stop - start}')
    model_params = pd.DataFrame(model.best_params_, index = ['random forest'])
    dirmake(f'data/part1/model{Model}/ML/random_forest')
    model_params.to_csv(f'data/part1/model{Model}/ML/random_forest/model_params.csv')
    log(f'ML - las losowy - nauczanie - start')
    start = datetime.now()
    model = RandomForestRegressor(**model.best_params_)
    model.fit(train_data, train_labels)
    stop = datetime.now()
    log(f'ML - las losowy - nauczanie - koniec {stop - start}')
    print(' --- ZAKOŃCZONO')
    path = f'data/part1/model{Model}/ML/random_forest/model.pk1'
    print('Zapisywanie modelu do pliku {} oraz jego parametrów'.format(path))
    save_model(model, path)
    print(' --- ZAKOŃCZONO')
    print('Testowanie modelu random forest...')
    log(f'ML - las losowy - przewidywanie - start')
    start = datetime.now()
    predicted_labels = model.predict(test_data)
    stop = datetime.now()
    log(f'ML - las losowy - przewidywanie - koniec {stop - start}')
    print(' --- ZAKOŃCZONO')
    print('Translacja przewidywań...')
    results = pd.DataFrame({'expo': test_label, 'expo_est': predicted_labels})
    print(' --- ZAKOŃCZONO')
    print('Zapisywanie wyników do pliku...')
    results.to_csv(f'data/part1/model{Model}/ML/random_forest/estimated.csv')
    print(' --- ZAKOŃCZONO')

def gradient_boosting(features, part, Model):
    train_data, train_labels, test_data, test_label = split_data(features, number_to_learn, part)
    hiperparam_data = train_data[:floor(number_to_learn/10)]
    hiperparam_labels = train_labels[:floor(number_to_learn/10)]
    print('Wyznaczanie modelu gradient boosting...')
    # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    learning_rate = [2**-4 * i for i in range(1, 15)]
    n_estimators = list(range(100,1001,100))
    max_depth = list(range(2,20,1))
    min_samples_split = list(range(2,11))
    min_samples_leaf = list(range(1,6))
    max_features = ['auto', 'sqrt']
    random_grid = {'learning_rate': learning_rate,
                   'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'max_features': max_features}
    model = GradientBoostingRegressor()
    log(f'ML - wzmocnienie gradientowe - szukanie superparametrów - start')
    start = datetime.now()
    model = RandomizedSearchCV(estimator = model,
                               param_distributions = random_grid,
                               n_iter = 30,
                               cv = 3,
                               verbose=2,
                               random_state = 42,
                               n_jobs = 3,
                               return_train_score=True,
                               refit = True)
    model.fit(hiperparam_data, hiperparam_labels)
    stop = datetime.now()
    log(f'ML - wzmocenienie gradientowe - szukanie superparametrów - koniec {stop - start}')
    model_params = pd.DataFrame(model.best_params_, index = ['gradient boosting'])
    dirmake(f'data/part1/model{Model}/ML/gradient_boosting')
    model_params.to_csv(f'data/part1/model{Model}/ML/gradient_boosting/model_params.csv')
    model = GradientBoostingRegressor(**model.best_params_)
    # params = pd.read_csv(f'data/part1/model{Model}/ML/gradient_boosting/model_params.csv', index_col='Unnamed: 0')
    # params = params.to_dict(orient = 'list')
    # for key,value in params.items():
    #     params[key] = value[0]
    # model = GradientBoostingRegressor(**params)
    log(f'ML - wzmocnienie gradientowe - nauczanie - start')
    start = datetime.now()
    model.fit(train_data, train_labels)
    stop = datetime.now()
    log(f'ML - wzmocenienie gradientowe - nauczanie - koniec {stop - start}')
    print(' --- ZAKOŃCZONO')
    path = f'data/part1/model{Model}/ML/gradient_boosting/model.pk1'
    print('Zapisywanie modelu do pliku {} oraz jego parametrów'.format(path))
    save_model(model, path)
    print(' --- ZAKOŃCZONO')
    print('Testowanie modelu gradient boosting...')
    log(f'ML - wzmocnienie gradientowe - przewidywanie - start')
    start = datetime.now()
    predicted_labels = model.predict(test_data)
    stop = datetime.now()
    log(f'ML - wzmocenienie gradientowe - przewidywanie - koniec {stop - start}')
    print(' --- ZAKOŃCZONO')
    print('Translacja przewidywań...')
    results = pd.DataFrame({'expo': test_label, 'expo_est': predicted_labels})
    print(' --- ZAKOŃCZONO')
    print('Zapisywanie wyników do pliku...')
    results.to_csv(f'data/part1/model{Model}/ML/gradient_boosting/estimated.csv')
    print(' --- ZAKOŃCZONO')