import tensorflow as tf 


def up_block(input):
    #
    x = tf.nn.leaky_relu(input, 0.2, 'lrelu')
    