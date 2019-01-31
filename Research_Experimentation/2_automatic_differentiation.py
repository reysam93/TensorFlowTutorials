#!/usr/bin/env python3

"""
Tutorial 2: Automatic differentiation and gradient tape
"""


import tensorflow as tf


def f(x, y):
  output = 1.0
  for i in range(y):
    if i > 1 and i < 5:
      output = tf.multiply(output, x)
  return output

def grad(x, y):
  with tf.GradientTape() as t:
    t.watch(x)
    out = f(x, y)
  return t.gradient(out, x)


if __name__ == "__main__":
    tf.enable_eager_execution()

    # Gradient tapes
    x = tf.ones((2,2))
    with tf.GradientTape() as t:
        t.watch(x)
        y = tf.reduce_sum(x)
        z = tf.multiply(y,y)

    dz_dx = t.gradient(z,x)
    for i in [0,1]:
        for j in [0,1]:
            assert dz_dx[i][j].numpy() == 8

    x = tf.ones((2,2))
    with tf.GradientTape() as t:
        t.watch(x)
        y = tf.reduce_sum(x)
        z = tf.multiply(y,y)

    dz_dy = t.gradient(z, y)
    assert dz_dy.numpy() == 8.0

    x = tf.constant(3.0)
    with tf.GradientTape(persistent=True) as t:
      t.watch(x)
      y = x * x
      z = y * y
    dz_dx = t.gradient(z, x)  # 108.0 (4*x^3 at x = 3)
    dy_dx = t.gradient(y, x)  # 6.0
    del t

    ## Recording control flow
    x = tf.convert_to_tensor(2.0)
    assert grad(x, 6).numpy() == 12.0
    assert grad(x, 5).numpy() == 12.0
    assert grad(x, 4).numpy() == 4.0

    ## High-order gradients
     x = tf.Variable(1.0)
     with tf.GradientTape() as t:
        with tf.GradientTape() as t2:
            y = x * x * x
        dy_dx = t2.gradient(y, x)
    d2y_dx2 = t.gradient(dy_dx, x)
    assert dy_dx.numpy() == 3.0
    assert d2y_dx2.numpy() == 6.0
