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
from global_params import part_to_learn
from sklearn.linear_model import LinearRegression
from pickle import dump, load
from generating_data import dirmake
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt

def split_data(dataset, prc_learn, part):
    train_dataset = dataset.sample(frac=0.8, random_state=0)
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

def linear_regression(features, part):
    train_data, train_labels, test_data, test_label = split_data(features, part_to_learn, part)
    print('Wyznaczanie modelu wilowymiarowej regresji liniowej...')
    model = LinearRegression(normalize = True, n_jobs=-1)
    model = model.fit(train_data, train_labels)
    print(' --- ZAKOŃCZONO')
    path = 'data/part1/ML/linear_regression/model.pk1'
    print('Zapisywanie modelu do pliku {}'.format(path))
    save_model(model, path)
    print(' --- ZAKOŃCZONO')
    print('Testowanie modelu wielowymiarowej regresji liniowej...')
    predicted_labels = model.predict(test_data)
    print(' --- ZAKOŃCZONO')
    print('Translacja przewidywań...')
    results = pd.DataFrame({'expo': test_label, 'expo_est': predicted_labels})
    print(' --- ZAKOŃCZONO')
    print('Translacja wyników do pliku...')
    results.to_csv('data/part1/ML/linear_regression/estimated.csv')
    print(' --- ZAKOŃCZONO')

def decision_tree(features, part):
    train_data, train_labels, test_data, test_label = split_data(features, part_to_learn, part)
    print('Wyznaczanie modelu wilowymiarowej regresji liniowej...')
    max_depth = list(range(5,56,5))
    min_samples_split = list(range(1,11))
    min_samples_leaf = list(range(1,6))
    max_features = ['auto', 'sqrt']
    random_grid = {'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'max_features': max_features}
    model = DecisionTreeRegressor()
    model = RandomizedSearchCV(estimator = model,
                               param_distributions = random_grid,
                               n_iter = 10,
                               cv = 10,
                               verbose=2,
                               random_state = 42,
                               n_jobs = -1,
                               return_train_score=True,
                               refit = True)
    model = model.fit(train_data, train_labels)
    model_params = pd.DataFrame(model.best_params_, index = ['decision tree'])
    model_params.to_csv('data/part1/ML/decision_tree/model_params.csv')
    model = DecisionTreeRegressor(**model.best_params_)
    model.fit(train_data, train_labels)
    print(' --- ZAKOŃCZONO')
    path = 'data/part1/ML/decision_tree/model.pk1'
    print('Zapisywanie modelu do pliku {}'.format(path))
    save_model(model, path)
    plt.cla()
    plt.figure(figsize=(10,10))
    plot_tree(model, max_depth = 3, feature_names = list(test_data), fontsize=10)
    path = 'data/part1/ML/decision_tree/model.png'
    plt.savefig(path, transparent = True, bbox_inches = 'tight')
    print(' --- ZAKOŃCZONO')
    print('Testowanie modelu wielowymiarowej regresji liniowej...')
    predicted_labels = model.predict(test_data)
    print(' --- ZAKOŃCZONO')
    print('Translacja przewidywań...')
    results = pd.DataFrame({'expo': test_label, 'expo_est': predicted_labels})
    print(' --- ZAKOŃCZONO')
    print('Translacja wyników do pliku...')
    results.to_csv('data/part1/ML/decision_tree/estimated.csv')
    print(' --- ZAKOŃCZONO')

def random_forest(features, part):
    train_data, train_labels, test_data, test_label = split_data(features, part_to_learn, part)
    print('Wyznaczanie modelu random forest...')
    # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    n_estimators = list(range(100,1001,100))
    max_depth = list(range(5,56,5))
    min_samples_split = list(range(1,11))
    min_samples_leaf = list(range(1,6))
    max_features = ['auto', 'sqrt']
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'max_features': max_features}
    model = RandomForestRegressor(bootstrap = True, max_samples=0.5)
    model = RandomizedSearchCV(estimator = model,
                               param_distributions = random_grid,
                               n_iter = 10,
                               cv = 10,
                               verbose=2,
                               random_state = 42,
                               n_jobs = -1,
                               return_train_score=True,
                               refit = True)
    model.fit(train_data, train_labels)
    print(model.best_params_)
    print(' --- ZAKOŃCZONO')
    path = 'data/part1/ML/random_forest/model.pk1'
    print('Zapisywanie modelu do pliku {} oraz jego parametrów'.format(path))
    save_model(model, path)
    model_params = pd.DataFrame(model.best_params_, index = ['random forest'])
    model_params.to_csv('data/part1/ML/random_forest/model_params.csv')
    print(' --- ZAKOŃCZONO')
    print('Testowanie modelu random forest...')
    predicted_labels = model.predict(test_data)
    print(' --- ZAKOŃCZONO')
    print('Translacja przewidywań...')
    results = pd.DataFrame({'expo': test_label, 'expo_est': predicted_labels})
    print(' --- ZAKOŃCZONO')
    print('Zapisywanie wyników do pliku...')
    results.to_csv('data/part1/ML/random_forest/estimated.csv')
    print(' --- ZAKOŃCZONO')
