#!/usr/bin/env python3

"""
Tutorial 1: Eager execution
"""

import tensorflow as tf
import numpy as np
import time
import tempfile


def time_matmul(x):
  start = time.time()
  for loop in range(10):
    tf.matmul(x, x)

  result = time.time()-start

  print("10 loops: {:0.2f}ms".format(1000*result))


if __name__ == "__main__":
    tf.enable_eager_execution()

    # Tensors
    ## Operations and tensor atributtes
    print(tf.add(1,2))
    print(tf.add([1,2],[3,4]))
    print(tf.square([1, 5]))
    print(tf.reduce_sum([1,2,3]))
    print(tf.encode_base64("hello world"))
    print(tf.square(2) + tf.square(3))
    print()

    x = tf.matmul([[1]],[[2,3]])
    print(x.shape)
    print(x.dtype)
    print()

    ## NumPy Compatibility
    ndarray = np.ones([3, 3])
    print("TensorFlow operations convert numpy arrays to Tensors automatically")
    tensor = tf.multiply(ndarray,42)
    print(tensor)
    print("And NumPy operations convert Tensors to numpy arrays automatically")
    print(np.add(tensor,1))
    print("The .numpy() method explicitly converts a Tensor to a numpy array")
    print(tensor.numpy())
    print()

    # GPU acceleration
    x = tf.random_uniform([3, 3])
    print("Is there a GPU available: ")
    print(tf.test.is_gpu_available())
    print("Is the Tensor on GPU #0:  "),
    print(x.device.endswith('GPU:0'))
    print()

    ## Explicit device placement
    print("On CPU:")
    with tf.device("CPU:0"):
        x = tf.random_uniform([1000,1000])
        assert x.device.endswith("CPU:0")
        time_matmul(x)

    if tf.test.is_gpu_available():
        with tf.device("GPU:0"):
            x = tf.random_uniform([1000, 1000])
            assert x.device.endswith("GPU:0")
            time_matmul(x)
    print()

    # Datasets
    ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
    _, filename = tempfile.mkstemp()
    with open(filename,'w') as f:
        f.write("""Line 1
Line 2
Line 3
""")
    ds_file = tf.data.TextLineDataset(filename)
    ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
    print('Elements of ds_tensors:')
    ds_file = ds_file.batch(2)
    for x in ds_tensors:
      print(x)

    print('\nElements in ds_file:')
    for x in ds_file:
      print(x)
