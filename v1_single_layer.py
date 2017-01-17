import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.optimizers import SGD

def network(placeholderX=None):

    x = input_data(shape=[None, 96, 96, 1], name='input', placeholder=placeholderX)

    x = fully_connected(x, 50, activation='relu', scope='fc1')

    x = fully_connected(x, 30, activation='linear', weights_init=tflearn.initializations.zeros(), scope='fc2')

    return x

def optimizer():
    return SGD(learning_rate=0.7, lr_decay=0.96, decay_step=2400)
