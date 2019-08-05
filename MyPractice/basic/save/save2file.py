import tensorflow as tf
import numpy as np

W = tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weights')
b = tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')

init = tf.global_variables_initializer()

saver = tf.train.Saver()
"""保存时, 首先要建立一个 tf.train.Saver() 用来保存, 提取变量.
 再创建一个名为my_net的文件夹, 用这个 saver 来保存变量到这个目录
 "my_net/save_net.ckpt"."""

with tf.Session() as sess:
     sess.run(init)
     save_path = saver.save(sess,"my_net/save_net.ckpt")
     print("Save to path:",save_path)





