# Seed value
# Apparently you may use different seed values at each stage
seed_value = 0

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)

import numpy as np
import tensorflow as tf
import random as python_random

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(123)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(123)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.random.set_seed(1234)

import seaborn as sns
import pandas as pd
import talos as ta
from numpy import zeros, newaxis
from matplotlib import pyplot as plt
from joblib import dump, load
from fast_ml.model_development import train_valid_test_split
from keras.utils.vis_utils import plot_model
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, max_error, mean_absolute_error
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, GRU, Bidirectional, SimpleRNN, Conv1D, MaxPooling1D, \
    Flatten, Activation

# Read Csv
file = r"C:\Users\lisof_ip7wdlf\Downloads\PG.csv"
df = pd.read_csv(file, parse_dates=['Date'], index_col='Date')
plt.style.use('seaborn')

# Let's say we want to split the data in 80:10:10 for train:valid:test dataset it was decided to use a manual
train_size = 0.8
valid_size = 0.1

train_index = int(len(df) * train_size)

df_train = df[0:train_index]
df_rem = df[train_index:]

valid_index = int(len(df) * valid_size)

df_valid = df[train_index:train_index + valid_index]
df_test = df[train_index + valid_index:]
test_index = df_test.shape[0]

X_train, y_train = df_train, df_train[['Close']]
X_valid, y_valid = df_valid, df_valid[['Close']]
X_test, y_test = df_test, df_test[['Close']]

print('X_train.shape:', X_train.shape, 'y_train.shape:', y_train.shape)
print('X_valid.shape:', X_valid.shape, 'y_valid.shape:', y_valid.shape)
print('X_test.shape:', X_test.shape, 'y_test.shape:', y_test.shape)

# Normalization
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

'''
# Other transformers
# StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# PowerTransformer
pt = PowerTransformer(method='box-cox')
X_train = pt.fit_transform(X_train)
X_valid = pt.transform(X_valid)
X_test = pt.transform(X_test)
'''

# Convert y sets to numpy array
y_train = y_train.to_numpy()
y_valid = y_valid.to_numpy()
y_test = y_test.to_numpy()


# Create a 3D input
def create_dataset(X, y, lag=1, n_ahead=1):
    Xs, ys = [], []
    for i in range(len(X) - lag - n_ahead):
        Xs.append(X[i:(i + lag)])
        ys.append(y[(i + lag):(i + lag + n_ahead)])
    return np.array(Xs), np.array(ys)


# Choose lag window
time_steps = 20
# Choose 1 for a single step prediction or 2, 3, ..., n for a multi step prediction
step_ahead = 1

X_train, y_train = create_dataset(X_train, y_train, time_steps, step_ahead)
X_test, y_test = create_dataset(X_test, y_test, time_steps, step_ahead)
X_valid, y_valid = create_dataset(X_valid, y_valid, time_steps, step_ahead)


# Create Simple RNN model
def create_simple_rnn():
    model = Sequential()
    model.add(SimpleRNN(35, return_sequences=True,
                        input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
    model.add(SimpleRNN(20, return_sequences=True, activation='relu'))
    model.add(SimpleRNN(5, return_sequences=True, activation='relu'))
    model.add(SimpleRNN(2, activation='relu'))
    model.add(Dropout(0))
    model.add(Dense(units=y_train.shape[1], activation='linear'))
    # Compile model
    model.compile(loss='mse', optimizer='adam')  # Default_lr = 0.001
    model.summary()

    return model


# Create GRU model
def create_gru():
    model = Sequential()
    model.add(GRU(32, return_sequences=True,
                  input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
    model.add(GRU(32, return_sequences=True, activation='relu'))
    model.add(GRU(32, return_sequences=True, activation='relu'))
    model.add(GRU(32, activation='relu'))
    model.add(Dropout(0))
    model.add(Dense(units=y_train.shape[1], activation='linear'))
    # Compile model
    model.compile(loss='mse', optimizer='adam')  # Default_lr = 0.001
    model.summary()

    return model


# Create GRU model
def create_lstm():
    model = Sequential()
    model.add(LSTM(100, return_sequences=True,
                   input_shape=(X_train.shape[1], X_train.shape[2]), activation='LeakyReLU'))
    model.add(LSTM(100, return_sequences=False, activation='LeakyReLU'))
    model.add(Dropout(0))
    model.add(Dense(units=y_train.shape[1], activation='linear'))
    # Compile model
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))  # Default_lr = 0.001
    model.summary()

    return model


def create_cnn_1d():
    # model.add(tf.keras.layers.Dense(1))
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu',
                     input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(Dropout(0))
    model.add(Conv1D(filters=11, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(Dropout(0))
    model.add(Conv1D(filters=21, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(Dropout(0))
    model.add(Flatten())
    model.add(Dense(units=y_train.shape[1]))
    model.add(Activation('linear'))
    # Compile model
    model.compile(loss='mse', optimizer='adam')  # Default_lr = 0.001
    model.summary()

    return model


# GRU and LSTM
# model_simple_rnn = create_simple_rnn()
# model_gru = create_gru()
model_lstm = create_lstm()
# model_cnn_1d = create_cnn_1d()


def fit_simple_rnn(model):
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
    history = model.fit(X_train, y_train,
                        epochs=70,
                        batch_size=350,
                        validation_data=[X_valid, y_valid],
                        callbacks=[early_stop])
    return history


def fit_gru(model):
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=100)
    history = model.fit(X_train, y_train,
                        epochs=35,
                        batch_size=60,
                        validation_data=[X_valid, y_valid],
                        callbacks=[early_stop])
    return history


def fit_lstm(model):
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
    history = model.fit(X_train, y_train,
                        epochs=100,
                        batch_size=200,
                        validation_data=[X_valid, y_valid],
                        callbacks=[early_stop])
    return history


def fit_cnn_1d(model):
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(X_train, y_train,
                        epochs=100,
                        batch_size=100,
                        validation_data=[X_valid, y_valid],
                        callbacks=[early_stop])
    return history


# history_simple_rnn = fit_simple_rnn(model_simple_rnn)
# history_gru = fit_gru(model_gru)
# history_lstm = fit_lstm(model_lstm)
# history_cnn_1d = fit_cnn_1d(model_cnn_1d)



y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])


# LOSS CURVE
# Plot train loss and validation loss

def plot_loss(history, model_name):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss curve_' + model_name, fontsize=16, y=1.01)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train loss', 'Validation loss'], loc='upper right')
    # plt.savefig('Loss curve_' + model name + '.png', dpi=1200)
    plt.show()


# plot_loss(history_simple_rnn, 'SimpleRNN')
# plot_loss(history_gru, 'GRU')
# plot_loss(history_lstm, 'LSTM')
# plot_loss(history_cnn_1d, 'CNN_1D')


# Make prediction
def prediction(model):
    print(X_test.shape)
    prediction = model.predict(X_test)
    return prediction


# prediction_simpleRNN = prediction(model_simple_rnn)
# prediction_gru = prediction(model_gru)
# prediction_lstm = prediction(model_lstm)
# prediction_cnn_1d = prediction(model_cnn_1d)


def plot_future(prediction, model_name, y_test):
    plt.figure(figsize=(10, 6))
    range_future = len(prediction)
    plt.plot(np.arange(range_future), np.array(y_test), label='True Future')
    plt.plot(np.arange(range_future), np.array(prediction), label='Prediction')
    plt.title('True future vs prediction for ' + model_name)
    plt.legend(loc='upper left')
    plt.xlabel('Time (day)')
    plt.ylabel('Stock Price (€)')
    # plt.savefig('Prediction_Evaluation_plot_' + model_name + '.png', dpi=1200)
    plt.show()


# plot_future(prediction_simpleRNN, 'SimpleRNN', y_test)
# plot_future(prediction_gru, 'GRU', y_test)
# plot_future(prediction_lstm, 'LSTM', y_test)
# plot_future(prediction_cnn_1d, 'CNN_1D', y_test)


# Define a function to calculate MAE and RSME
def evaluate_prediction(predicted, actual, model_name):
    if step_ahead == 1:
        rsme = np.sqrt((mean_squared_error(predicted, actual)))
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        max_err = max_error(actual, predicted)
        print(model_name + ' performance:')
        print('R^2: {:.4f} %'.format(r2 * 100))
        print('Mean Absolute Error: {:.4f}'.format(mae))
        print('Root Mean Square Error: {:.4f}'.format(rsme))
        print('Max_error: {:.4f}'.format(max_err))
        print('')
        return
    else:
        titles = ["RMSE", "MAE", "R^2"]
        # calculate an RMSE score for each day
        # calculate mse
        rmse = np.sqrt(mean_squared_error(predicted, actual, multioutput='raw_values'))
        mae = mean_absolute_error(predicted, actual, multioutput='raw_values')
        r2 = r2_score(predicted, actual, multioutput='raw_values')
        df_scores = pd.DataFrame(list(zip(rmse, mae, r2)), columns=[f'{x}' for x in titles])
        df_scores.index += 1

        colors = plt.rcParams["axes.prop_cycle"]()
        a = 1  # number of rows
        b = 3  # number of columns
        c = 1  # initialize plot counter
        fig = plt.figure(figsize=(15, 6))
        for i in titles:
            plt.subplot(a, b, c)
            plt.title(f'{i}')
            next_colour = next(colors)["color"]
            df_scores[f'{i}'].plot(marker='o', color=next_colour)
            plt.xticks((range(0, df_scores.shape[0] + 1)))
            plt.legend(loc='upper left')
            plt.xlabel('Forecast Range (Day)')
            plt.ylabel(f'{i}')
            c = c + 1

        plt.subplots_adjust(.5)
        fig.suptitle("Evaluation of performances' trend in the multi step forecasted range", fontsize=16, y=1)
        plt.tight_layout()
        # plt.savefig('EvaluationMultiplePrediction_PG.png', dpi=1200)
        plt.show()

        # calculate overall RMSE
        overall_rmse = np.sqrt(mean_squared_error(predicted, actual, multioutput='uniform_average'))
        overall_mae = mean_absolute_error(predicted, actual, multioutput='uniform_average')
        overall_r2 = r2_score(predicted, actual, multioutput='uniform_average')
        print(model_name + ' performance:')
        print('R^2: {:.4f} %'.format(overall_r2 * 100))
        print('Mean Absolute Error: {:.4f}'.format(overall_mae))
        print('Root Mean Square Error: {:.4f}'.format(overall_rmse))
        print('')
        return


# evaluate_prediction(prediction_simpleRNN, y_test, 'SimpleRNN')
# evaluate_prediction(prediction_gru, y_test, 'GRU')
# evaluate_prediction(prediction_lstm, y_test, 'LSTM')
# evaluate_prediction(prediction_cnn_1d, y_test, 'CNN_1D')


def save_model(model, model_name):
    model.save('./best_model_simple_rnn_single_step')
    plot_model(model, to_file='model_' + model_name + '.png', show_shapes=True)
    return


# save_model(model_simple_rnn)
# save_model(model_gru)
# save_model(model_lstm)
# save_model(model_cnn_1d)


'''
loaded_model = tf.keras.models.load_model(r"")
loaded_model.summary()

prediction_gru12 = prediction(loaded_model)
print(prediction_gru12.shape)

plot_future(prediction_gru12, 'LSTM', y_test)
evaluate_prediction(prediction_gru12, y_test, 'LSTM')
'''