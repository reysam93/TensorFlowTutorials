#!/usr/bin/env python3
"""
Tutorial 2: Text Clasification
It classifies movie reviews as positive or negative.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def decode_review(reverse_word_index, text):
    return ' '.join([reverse_word_index.get(i,'?') for i in text])


def build_model():
    # Input shape is the vocabulary count for the moview reviews
    vocab_size= 10000
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
    return model


def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc)+1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label="Training loss")
    # b is for solid blue line
    plt.plot(epochs, val_loss, 'b', label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure() # clear figure
    plt.plot(epochs, acc, 'bo', label="Training acc")
    plt.plot(epochs, val_acc, 'b', label="Validation acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Download the IMDB dataset
    imdb = keras.datasets.imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    # Explore the data
    print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
    print(train_data.shape)
    # Create decode dict
    word_index = imdb.get_word_index()
    word_index = {k:(v+3) for k,v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    print("Decoded review:\n", decode_review(reverse_word_index, train_data[0]))

    # Prepare the data
    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=word_index["<PAD>"],
                                                            padding="post",
                                                            maxlen=256)
    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                            value=word_index["<PAD>"],
                                                            padding="post",
                                                            maxlen=256)
    print("Decoded padded review:\n", decode_review(reverse_word_index, train_data[0]))

    # Build and compile the models
    model = build_model()
    model.summary()
    model.compile(optimizer=tf.train.AdamOptimizer(),
                    loss='binary_crossentropy',
                    metrics=["accuracy"])

    # Crate validation set
    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]
    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]

    # Train the model
    history = model.fit(partial_x_train, partial_y_train, epochs=40,
                        batch_size=512, validation_data=(x_val, y_val), verbose=1)

    # Evaluate the model
    results = model.evaluate(test_data, test_labels)
    print(results)

    # Plot accuracy and loss over time
    plot_history(history)
