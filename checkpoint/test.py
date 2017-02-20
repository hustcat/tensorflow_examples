import os

import tensorflow as tf
import numpy as np

isTrain = False
train_steps = 100
checkpoint_steps = 50
checkpoint_dir = './chk_dir'

x = tf.placeholder(tf.float32, shape=[None, 1])
y = 4 * x + 4

w = tf.Variable(tf.random_normal([1], -1, 1))
b = tf.Variable(tf.zeros([1]))
y_predict = w * x + b


loss = tf.reduce_mean(tf.square(y - y_predict))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
x_data = np.reshape(np.random.rand(10).astype(np.float32), (10, 1))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if isTrain:
        for i in xrange(train_steps):
            sess.run(train, feed_dict={x: x_data})
            if (i + 1) % checkpoint_steps == 0:
                checkpoint_file = os.path.join(checkpoint_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=i+1)
    else:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            pass
        print(sess.run(w))
        print(sess.run(b))
