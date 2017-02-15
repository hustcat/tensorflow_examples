import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()

arr = np.array([[31, 23,  4, 24, 27, 34],
                [18,  3, 25,  0,  6, 35],
                [28, 14, 33, 22, 20,  8],
                [13, 30, 21, 19,  7,  9],
                [16,  1, 26, 32,  2, 29],
                [17, 12,  5, 11, 10, 15]])

# arr has rank 2 and shape (6, 6)
print(tf.rank(arr).eval())

print(tf.shape(arr).eval())

print(tf.argmax(arr, 0).eval())
