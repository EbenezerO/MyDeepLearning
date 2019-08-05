import tensorflow as tf
import numpy as np
"""restore variable
redefine the same shape and same type for your variables
"""

W = tf.Variable(np.arange(6).reshape(2,3),name='weights',dtype=tf.float32)


b = tf.Variable(np.arange(3).reshape(1,3),name='biases',dtype=tf.float32)

# not need init step
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,"my_net/save_net.ckpt")
    print("weights:",sess.run(W))
    print("biases:",sess.run(b))