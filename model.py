import math

import tensorflow as tf

from constants import IMAGE_PIXELS, OUTPUT

def inference(images, hidden1_units, hidden2_units):
	with tf.name_scope('layer1'):
		# Placeholder Model
		with tf.name_scope('weights'):
			W = tf.Variable(tf.zeros([96 * 96, 30]))
		with tf.name_scope('biases'):
			b = tf.Variable(tf.zeros([30]))
		with tf.name_scope('Wx_plus_b'):
			y = tf.matmul(images, W) + b
		return y

def loss(y, y_):
	# y - prediction
	# y_ - ground truth
	with tf.name_scope('loss'):
		return tf.reduce_mean(tf.square(y - y_))


def training(loss, learning_rate):
	with tf.name_scope('train'):
		# Create the gradient descent optimizer with the given learning rate.
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		# Create a variable to track the global step.
		global_step = tf.Variable(0, name='global_step', trainable=False)
		# Use the optimizer to apply the gradients that minimize the loss
		# (and also increment the global step counter) as a single training step.
		train_op = optimizer.minimize(loss, global_step=global_step)
		return train_op


# Best guess for leaderboard score
def evaluation(y, y_):
	# y - prediction
	# y_ - ground truth
	with tf.name_scope('eval'):
		mse = tf.reduce_mean(tf.square(y - y_))
		return mse#math.sqrt(mse) * 48

