import math

import tensorflow as tf

from constants import IMAGE_PIXELS, OUTPUT

# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    #initial = tf.zeros(shape)
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    #initial = tf.zeros(shape)
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations


def inference(x, keep_prob):

	hidden1 = nn_layer(x, IMAGE_PIXELS, 50, 'layer1')

	with tf.name_scope('dropout'):
		dropped = tf.nn.dropout(hidden1, keep_prob)
    
	hidden2 = nn_layer(dropped, 50, OUTPUT, 'layer2', act=tf.identity)

	return hidden2

def loss(y, y_):
	# y - prediction
	# y_ - ground truth
	with tf.name_scope('loss'):
		return tf.reduce_mean(tf.square(y - y_))

def training(loss, learning_rate):
	with tf.name_scope('train'):
		# Create the gradient descent optimizer with the given learning rate.
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		#optimizer = tf.train.AdagradOptimizer(1e-2)
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

