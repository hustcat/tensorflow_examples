import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()

a = tf.constant([10, 20])
b = tf.constant([1.0, 2.0])
c = tf.add(a, b)
v = sess.run(c)
print(v)
