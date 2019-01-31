#!/usr/bin/env python3

"""
Tutorial 5: Save and restore models
"""

from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras


def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.keras.activations.softmax)
    ])
    model.compile(optimizer='adam',
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
    return model


if __name__ == "__main__":
    # Setup
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_labels = train_labels[:1000]
    test_labels = test_labels[:1000]

    train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
    test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

    model = create_model()
    model.summary()

    # Save checkpoints during training
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    model.fit(train_images, train_labels,  epochs = 10,
              validation_data = (test_images,test_labels),
              callbacks = [cp_callback])

    model = create_model()
    loss, acc = model.evaluate(test_images, test_labels)
    print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

    model.load_weights(checkpoint_path)
    loss, acc = model.evaluate(test_images, test_labels)
    print("Restored model (automaticaly saved), accuracy: {:5.2f}%".format(100*acc))

    # include the epoch in the file name. (uses `str.format`)
    checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
                        checkpoint_path, verbose=1, save_weights_only=True,
                        # Save weights, every 5-epochs.
                        period=5)
    model = create_model()
    model.save_weights(checkpoint_path.format(epoch=0))
    model.fit(train_images, train_labels,
              epochs = 50, callbacks = [cp_callback],
              validation_data = (test_images,test_labels),
              verbose=0)
    model = create_model()
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)
    loss, acc = model.evaluate(test_images, test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))


    # Manually save weights
    model.save_weights('./checkpoints/my_checkpoint')
    model = create_model()
    model.load_weights('./checkpoints/my_checkpoint')
    loss,acc = model.evaluate(test_images, test_labels)
print("Restored model (manually saved), accuracy: {:5.2f}%".format(100*acc))
