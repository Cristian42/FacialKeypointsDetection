import tensorflow as tf

# Model
x = tf.placeholder(tf.float32, [None, 96 * 96])
W = tf.Variable(tf.zeros([96 * 96, 30]))
b = tf.Variable(tf.zeros([30]))
y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, 30])
mse = tf.reduce_mean(tf.square(y - y_))


