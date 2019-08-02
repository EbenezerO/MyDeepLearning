import tensorflow as tf
"""TF variable"""

# 定义变量，给定初始值和name
state = tf.Variable(0, name="counter")
# counter:0
print(state.name)

one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 这里只是定义，必须用session.run来执行
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))