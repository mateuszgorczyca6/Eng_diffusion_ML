from numpy import mean
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def test_model(predicted_values, real_values, model_name):
    N = len(predicted_values)
    ## R^2
    u, v = 0, 0
    v_mean = mean(real_values)
    ## eps = 0.05
    counter5 = 0
    ## eps = 0.025
    counter25 = 0
    ## max_error
    max_mistake = 0
    errors = []
    for i in range(N):
        ## R^2
        u += (real_values[i] - predicted_values[i]) ** 2
        v += (real_values[i] - v_mean) ** 2
        ## eps = 0.05
        if abs(real_values[i] - predicted_values[i] < 0.05):
            counter5 += 1
        ## eps = 0.025
        if abs(real_values[i] - predicted_values[i] < 0.025):
            counter25 += 1
        ## max_error
        if abs(real_values[i] - predicted_values[i]) > max_mistake:
            max_mistake = abs(real_values[i] - predicted_values[i])
        errors.append(real_values[i] - predicted_values[i])
    ## R^2
    R2 = 1 - u/v
    strength = pd.DataFrame({'R^2': R2, 
                             'eps = 0.05': counter5/N,
                             'eps = 0.025': counter25/N,
                             'max_error': max_mistake},
                            index = [model_name])
    # plt.plot(real_values, predicted_values, '.', alpha = 0.25)
    sns.displot({'r': real_values, 'e': predicted_values}, x = 'r', y = 'e', cbar=True)
    plt.plot([0,2], [0,2], color = 'black')
    plt.title(model_name)
    plt.show()
    return strength, errors


def dojob(path, name):
    data = pd.read_csv(path, index_col = 'Unnamed: 0')
    est_data = list(data['expo_est'])
    real_data = list(data['expo'])
    return test_model(est_data, real_data, name)

err = {}
table, er = dojob('data/part1/modelA/TAMSD/estimated.csv', 'TAMSD')
ttable = table
err['TAMSD'] = er
table, er = dojob('data/part1/modelA/ML/linear_regression/estimated.csv', 'linear regression')
ttable = pd.concat([ttable, table])
err['linear regression'] = er
table, er = dojob('data/part1/modelA/ML/decision_tree/estimated.csv', 'decision tree')
ttable = pd.concat([ttable, table])
err['decision tree'] = er
table, er = dojob('data/part1/modelA/ML/random_forest/estimated.csv', 'random forest')
ttable = pd.concat([ttable, table])
err['random forest'] = er
table, er = dojob('data/part1/modelA/ML/gradient_boosting/estimated.csv', 'gradient boosting')
ttable = pd.concat([ttable, table])
err['gradient boosting'] = er

print(ttable)
box = plt.boxplot(labels = err.keys(), x = err.values(), patch_artist=True, notch = True, vert = True)
plt.title('boxplot')
colors = ['lightblue', 'orange', 'lightgreen', 'pink', 'violet']
for path, color in zip(box['boxes'], colors):
    path.set_facecolor(color)
plt.show()


sns.displot(err, common_norm=False, stat="density")
plt.plot([0,0], [0,1.3], color = 'black')

plt.show()

sns.displot(err, kind = "kde", common_norm=False, bw_adjust = 0.7, fill=True)
plt.plot([0,0], [0,1.3], color = 'black')

plt.show()


sns.displot(err, kind = "kde", common_norm=False, bw_adjust=2)
plt.plot([0,0], [0,1.3], color = 'black')

plt.show()


sns.displot(err, kind = "ecdf")
plt.plot([0,0], [0,1.3], color = 'black')

plt.show()
