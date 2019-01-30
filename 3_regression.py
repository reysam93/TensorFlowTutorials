#!/usr/bin/env python3

"""
Tutorial 3: Predict house prices: regression
"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

def build_model(train_data):
    model = keras.Sequential([
                            keras.layers.Dense(64, activation=tf.nn.relu,
                            input_shape=(train_data.shape[1],)),
                            keras.layers.Dense(64, activation=tf.nn.relu),
                            keras.layers.Dense(1)
  ])

    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
    return model


def plot_history(history):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Abs Error [1000$]")
    plt.plot(history.epoch, np.array(history.history["mean_absolute_error"]),
            label="Train Loss")
    plt.plot(history.epoch, np.array(history.history["val_mean_absolute_error"]),
            label="Val Loss")
    plt.legend()
    plt.ylim([0, 5])
    plt.show()

EPOCHS = 500

if __name__ == "__main__":
    # Get dataset
    boston_housing = keras.datasets.boston_housing
    (train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()
    order = np.argsort(np.random.random(train_labels.shape))
    train_data = train_data[order]
    train_labels = train_labels[order]

    print("Training set: {}".format(train_data.shape))
    print("eTst set: {}".format(test_data.shape))

    # Print first rows for checking scale difference of the atributes
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                    'TAX', 'PTRATIO', 'B', 'LSTAT']
    df = pd.DataFrame(train_data, columns=column_names)
    df.head()

    # Chack that "labels" now are the price of houses in thousands of dollars
    print("First 10 labels:", train_labels[0:10])

    # Normalize features
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    print("Normalized train data:", train_data[0])

    # Create the model
    model = build_model(train_data)
    model.summary()

    # Train the model
    history = model.fit(train_data, train_labels, epochs=EPOCHS,
                        validation_split=0.2, verbose=0, callbacks=[PrintDot()])
    plot_history(history)

    # Same model with early stoping
    model = build_model(train_data)
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)
    history = model.fit(train_data, train_labels, epochs=EPOCHS,
                        validation_split=0.2, verbose=0,
                        callbacks=[early_stop, PrintDot()])
    [loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
    print("Testing set Mean Abs Error: ${:7.2f}".format(mae*1000))

    # Predict
    test_predictions = model.predict(test_data).flatten()

    plt.figure()
    plt.scatter(test_labels, test_predictions)
    plt.xlabel("True Values [1000$]")
    plt.ylabel("Predictions [1000$]")
    plt.axis("equal")
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    _ = plt.plot([-100,100], [-100, 100])
    error = test_predictions - test_labels
    plt.show()
    plt.figure()
    plt.hist(error, bins=50)
    plt.xlabel("Prediction Error [1000$]")
    _ = plt.ylabel("Count")
    plt.show()
