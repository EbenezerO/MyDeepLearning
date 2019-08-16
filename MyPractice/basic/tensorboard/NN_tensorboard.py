import tensorflow as tf
import numpy as np
"""
使用Tensorboard，我们首先要定义变量的命名空间name_scope，只有定义了name_scope，
我们在tensorboard中的Graph才会看起来井然有序。
所以，我们以修改一层网络的函数为例，来看一下如何使用name_scope，name_scope对神经网络的训练过程是没有影响的。
"""
def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    layer_name = "layer%s" % n_layer
    print(layer_name)
    with tf.name_scope(layer_name):
        with tf.name_scope("Weights"):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
            #概率分布的形式
            tf.summary.histogram(layer_name+'/weights',Weights)
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.add(tf.matmul(inputs,Weights),biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs


x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

#None表示给多少个example都可以
with tf.name_scope("input"):
    xs = tf.placeholder(tf.float32,[None,1],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')

l1 = add_layer(xs,1,10,1,activation_function=tf.nn.relu)
prediction = add_layer(l1,10,1,2,activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                         reduction_indices=[1]))
    tf.summary.scalar("loss",loss)


with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 1.2之前 tf.train.SummaryWriter("logs/",sess.graph)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logs/',sess.graph,)
    sess.run(init)
    for i in range(1000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i % 50 == 0:
            result = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
            writer.add_summary(result,i)

"""
我们使用命令来查看最后的结果：

tensorboard --logdir=E:\推荐系统\DeepLearning\MyPractice\basic\tensorboard\logs --host=127.0.0.1  注意检查路径 需要完整路径

然后我们就可以根据它提示的网址去访问我们的结果啦，用chrome是可以的呦。或者直接在浏览器做输入 localhost:6006

"""