import tensorflow as tf

def discriminator(inputs, channels=64, is_training=True):

    def conv2(batch_input, num_outputs=64, kernel_size=[3, 3], stride=1, norm=True, scope=None):
        return tf.contrib.layers.conv2d(batch_input, num_outputs, kernel_size, stride, 'SAME', 'NHWC', scope=scope,
               activation_fn=tf.nn.leaky_relu,
               normalizer_fn=tf.contrib.layers.instance_norm if norm else None,
               normalizer_params={'center': False, 'scale': False, 'data_format': 'NHWC'},
               weights_initializer=tf.variance_scaling_initializer(scale=2.0))

    with tf.device('/gpu:0'):
        net = conv2(inputs, channels, kernel_size=[5, 5], stride=2, norm=False, scope='input_stage')

        net = conv2(net, channels*2, stride=2, scope='disblock_1')

        net = conv2(net, channels*4, stride=2, scope='disblock_2')

        net = conv2(net, channels*8, stride=2, scope='disblock_3')

        net = tf.layers.flatten(net)

        with tf.variable_scope('dense_layer_1'):
            net = tf.layers.dense(net, channels*16, activation=tf.nn.leaky_relu, use_bias=False,
                                  kernel_initializer=tf.variance_scaling_initializer(scale=2.0))

        with tf.variable_scope('dense_layer_2'):
            net = tf.layers.dense(net, 1, use_bias=False)

    return net

