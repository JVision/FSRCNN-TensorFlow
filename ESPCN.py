import tensorflow as tf

class ESPCN(object):

  def __init__(self, config):
    self.name = "ESPCN"
    self.model_params = [64, 32] #[64, 32, 28]
    self.scale = config.scale
    self.radius = config.radius
    self.padding = config.padding
    self.images = config.images
    self.batch = config.batch
    self.label_size = config.label_size
    self.c_dim = config.c_dim
    self.weights, self.biases, self.alphas = {}, {}, {}

  def variable_summaries(self, var):
      """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
      with tf.name_scope('summaries'):
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

  def model(self):
    d = self.model_params
    m = len(d) + 2

    with tf.name_scope("Feature"):
        with tf.name_scope('weights'):
            size = self.padding + 1
            self.weights['w1'] = tf.get_variable('w1', initializer=tf.random_normal([size, size, 1, d[0]], stddev=0.0378, dtype=tf.float32))
            self.variable_summaries(self.weights['w1'])
        with tf.name_scope('biases'):
            self.biases['b1'] = tf.get_variable('b1', initializer=tf.zeros([d[0]]))
            self.variable_summaries(self.biases['b1'])
        with tf.name_scope('Wx_plus_b'):
            conv = tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='VALID') + self.biases['b1']
            tf.summary.histogram('pre_activations', conv)
        conv = self.prelu(conv, 1)
        tf.summary.histogram('activations', conv)

    for i in range(3, m):
        with tf.name_scope("Mapping{}".format(i-2)):
            with tf.name_scope('weights'):
                weights = tf.get_variable('w{}'.format(i), initializer=tf.random_normal([3, 3, d[i-3], d[i-2]], stddev=0.1179, dtype=tf.float32))
                self.variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = tf.get_variable('b{}'.format(i), initializer=tf.zeros([d[i-2]]))
                self.variable_summaries(biases)
            self.weights['w{}'.format(i)], self.biases['b{}'.format(i)] = weights, biases
            with tf.name_scope('Wx_plus_b'):
                conv = tf.nn.conv2d(conv, weights, strides=[1,1,1,1], padding='SAME') + biases
                tf.summary.histogram('pre_activations', conv)
            conv = self.prelu(conv, i)
            tf.summary.histogram('activations', conv)

    with tf.name_scope("Deconvolution"):
        with tf.name_scope('weights'):
            deconv_size = self.radius * self.scale * 2 + 1
            deconv_weights = tf.get_variable('w{}'.format(m + 1), initializer=tf.random_normal([deconv_size, deconv_size, 1, d[-1]], stddev=0.0001, dtype=tf.float32))
            self.variable_summaries(deconv_weights)
        with tf.name_scope('biases'):
            deconv_biases = tf.get_variable('b{}'.format(m + 1), initializer=tf.zeros([1]))
            self.variable_summaries(deconv_biases)
        self.weights['w{}'.format(m + 1)], self.biases['b{}'.format(m + 1)] = deconv_weights, deconv_biases
        with tf.name_scope('Wx_plus_b'):
            deconv_output = [self.batch, self.label_size, self.label_size, self.c_dim]
            deconv_stride = [1,  self.scale, self.scale, 1]
            deconv = tf.nn.conv2d_transpose(conv, deconv_weights, output_shape=deconv_output, strides=deconv_stride, padding='SAME') + deconv_biases
            tf.summary.histogram('output', deconv)

    return deconv

  def prelu(self, _x, i):
    """
    PreLU tensorflow implementation
    """
    alphas = tf.get_variable('alpha{}'.format(i), _x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    self.alphas['alpha{}'.format(i)] = alphas
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg
