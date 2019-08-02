import tensorflow as tf
import numpy as np


def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.add(tf.matmul(inputs,Weights),biases)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#创建训练集
x_data = np.linspace(-1,1,300)[:,np.newaxis] #x_data.shape=[300,1]
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise
#print(x_data,y_data)

#None表示给多少个example都可以,两者等价
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[300,1])

#创建模型
layer1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction = add_layer(layer1,10,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))

train_model = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train_model,feed_dict={xs:x_data,ys:y_data})
        if i % 100 == 0:
            print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))