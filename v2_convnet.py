import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.optimizers import SGD

def network(placeholderX=None):

    x = input_data(shape=[None, 96, 96, 1], name='input', placeholder=placeholderX)

    x = conv_2d(x, 32, 3, activation='relu', scope='conv1_1')
    x = max_pool_2d(x, 2, name='maxpool1')
    
    x = conv_2d(x, 64, 3, activation='relu', scope='conv2_1')
    x = conv_2d(x, 64, 3, activation='relu', scope='conv2_2')
    x = max_pool_2d(x, 2, name='maxpool2')

    x = fully_connected(x, 512, activation='relu', scope='fc3')
    x = dropout(x, 0.5, name='dropout1')

    x = fully_connected(x, 30, activation='linear', weights_init=tflearn.initializations.zeros(), scope='fc4')

    return x

def optimizer():
    return SGD(learning_rate=0.7, lr_decay=0.96, decay_step=2400)
