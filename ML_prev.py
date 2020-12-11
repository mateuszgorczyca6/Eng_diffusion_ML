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
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from generating_data import read_ML_features
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import plot_model
#
# # # Data preprocessing # # #
#
dataset = read_ML_features(1)
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('alpha')
test_labels = test_features.pop('alpha')

normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))
#
# # # Regresja liniowa # # #
#
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])
linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

linear_model.summary()

history = linear_model.fit(
    train_features, train_labels, 
    epochs=100,
    # suppress logging
    verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  plt.show()

plot_loss(history)

test_results = {}

test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=0)
#
# # # Model DNN # # #
#
def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

history = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

plot_loss(history)

test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)

results = pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T

print(results)
#
# # # testowanie predykcji # # #
#
test_predictions = dnn_model.predict(test_features).flatten()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 7]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')
plt.show()

plot_model(linear_model, to_file='linear_model.png', show_shapes=True, expand_nested=True)
plot_model(dnn_model, to_file='dnn_model.png', show_shapes=True, expand_nested=True)
#
# # # Random Forest
#
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('data/part1/ML/features.csv') # Other type of file could be used which contains tabular data

# Target column must be last to work below all cell's code correctly, If you don't have your target colum last then make necessary changes to below two lines of code


# Do required transformation(s) for X and/or y (If required)

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('alpha')
test_labels = test_features.pop('alpha')

regressor = RandomForestRegressor(n_estimators=50, random_state=0, ).fit(train_features, train_labels)

y_pred = regressor.predict(test_features)

r2_score(test_labels, y_pred)

plt.plot(test_labels, y_pred,'.')
plt.plot([0,2], [0,2])
plt.show()
#
# # # Polylinear regression
#
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

model = LinearRegression(normalize = True, n_jobs=-1)
model = model.fit(train_features, train_labels)
y_pred = model.predict(test_features)
print(y_pred)
r2_score(y_pred, test_labels)
plt.plot(y_pred, test_labels,'.')
plt.plot([0,2],[0,2])
plt.show()
