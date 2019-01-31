#!/usr/bin/env python3

"""
Tutorial 3: Custom training: basics
"""

import tensorflow as tf
import matplotlib.pyplot as plt


TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000


## Define the model
class Model(object):
    def __init__(self):
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self,x):
        return self.W * x + self.b

## Define loss function
def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))

## Define training loop
def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)


if __name__ == "__main__":
    tf.enable_eager_execution()

    # Variables
    ## Using Python state
    x = tf.zeros([10,10])
    x += 2
    print(x)

    ## Using TF variables
    v = tf.Variable(1.0)
    assert v.numpy() == 1.0
    v.assign(3.0)
    assert v.numpy() == 3.0
    v.assign(tf.square(v))
    assert v.numpy() == 9

    # Example: Fitting a linear model
    model = Model()
    assert model(3.0).numpy() == 15

    ## Obtaining training data
    inputs  = tf.random_normal(shape=[NUM_EXAMPLES])
    noise   = tf.random_normal(shape=[NUM_EXAMPLES])
    outputs = inputs * TRUE_W + TRUE_b + noise;
    plt.scatter(inputs, outputs, c='b')
    plt.scatter(inputs, model(inputs), c='r')
    plt.show()
    print('Current loss: '),
    print(loss(model(inputs), outputs).numpy())

    Ws, bs = [],[]
    epochs = range(10)
    for epoch in epochs:
        Ws.append(model.W.numpy())
        bs.append(model.b.numpy())
        current_loss = loss(model(inputs), outputs)
        train(model, inputs, outputs, learning_rate=0.1)
        print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %
            (epoch, Ws[-1], bs[-1], current_loss))

    ## Plot it
    plt.plot(epochs, Ws, 'r',
             epochs, bs, 'b')
    plt.plot([TRUE_W] * len(epochs), 'r--',
             [TRUE_b] * len(epochs), 'b--')
    plt.legend(['W', 'b', 'true W', 'true_b'])
    plt.show()
