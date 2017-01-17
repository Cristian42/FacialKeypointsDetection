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

    x = conv_2d(x, 128, 3, activation='relu', scope='conv3_1')
    x = conv_2d(x, 128, 3, activation='relu', scope='conv3_2')
    x = max_pool_2d(x, 2, name='maxpool3')
    
    x = conv_2d(x, 256, 3, activation='relu', scope='conv4_1')
    x = conv_2d(x, 256, 3, activation='relu', scope='conv4_2')
    x = max_pool_2d(x, 2, name='maxpool4')
    
    x = fully_connected(x, 1000, activation='relu', scope='fc5')
    x = dropout(x, 0.5, name='dropout1')

    x = fully_connected(x, 500, activation='relu', scope='fc6')
    x = dropout(x, 0.5, name='dropout2')
    
    x = fully_connected(x, 500, activation='relu', scope='fc7')

    x = fully_connected(x, 30, activation='linear', weights_init=tflearn.initializations.zeros(), scope='fc8')

    return x

def optimizer():
    return SGD(learning_rate=0.7, lr_decay=0.96, decay_step=2400)
