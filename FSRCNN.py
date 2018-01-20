import tensorflow as tf
from utils import tf_ssim

class Model(object):

  def __init__(self, config):
    self.name = "FSRCNN"
    # Different model layer counts and filter sizes for FSRCNN vs FSRCNN-s (fast), (d, s, m) in paper
    model_params = [[56, 12, 4], [32, 8, 1]]
    self.model_params = model_params[config.fast]
    self.scale = config.scale
    self.radius = config.radius
    self.padding = config.padding
    self.images = config.images
    self.batch = config.batch
    self.label_size = config.label_size
    self.c_dim = config.c_dim

  def model(self):

    d, s, m = self.model_params

    # Feature Extraction
    size = self.padding + 1
    weights = tf.get_variable('w1', shape=[size, size, 1, d], initializer=tf.variance_scaling_initializer())
    biases = tf.get_variable('b1', initializer=tf.zeros([d]))
    conv = tf.nn.conv2d(self.images, weights, strides=[1,1,1,1], padding='VALID', data_format='NHWC')
    conv = self.prelu(tf.nn.bias_add(conv, biases, data_format='NHWC'), 1)

    # Shrinking
    if self.model_params[1] > 0:
      weights = tf.get_variable('w2', shape=[1, 1, d, s], initializer=tf.variance_scaling_initializer())
      biases = tf.get_variable('b2', initializer=tf.zeros([s]))
      conv = tf.nn.conv2d(conv, weights, strides=[1,1,1,1], padding='SAME', data_format='NHWC')
      conv = self.prelu(tf.nn.bias_add(conv, biases, data_format='NHWC'), 2)
    else:
      s = d

    # Mapping (# mapping layers = m)
    for i in range(3, m + 3):
      weights = tf.get_variable('w{}'.format(i), shape=[3, 3, s, s], initializer=tf.variance_scaling_initializer())
      biases = tf.get_variable('b{}'.format(i), initializer=tf.zeros([s]))
      conv = tf.nn.conv2d(conv, weights, strides=[1,1,1,1], padding='SAME', data_format='NHWC')
      conv = self.prelu(tf.nn.bias_add(conv, biases, data_format='NHWC'), i)

    # Expanding
    if self.model_params[1] > 0:
      expand_weights = tf.get_variable('w{}'.format(m + 3), shape=[1, 1, s, d], initializer=tf.variance_scaling_initializer())
      expand_biases = tf.get_variable('b{}'.format(m + 3), initializer=tf.zeros([d]))
      conv = tf.nn.conv2d(conv, expand_weights, strides=[1,1,1,1], padding='SAME', data_format='NHWC')
      conv = self.prelu(tf.nn.bias_add(conv, expand_biases, data_format='NHWC'), m + 3)

    # Deconvolution
    deconv_size = self.radius * self.scale * 2 + 1
    deconv_weights = tf.get_variable('w{}'.format(m + 4), shape=[deconv_size, deconv_size, 1, d], initializer=tf.variance_scaling_initializer(scale=0.01))
    deconv_biases = tf.get_variable('b{}'.format(m + 4), initializer=tf.zeros([1]))
    deconv_output = [self.batch, self.label_size, self.label_size, self.c_dim]
    deconv_stride = [1,  self.scale, self.scale, 1]
    deconv = tf.nn.conv2d_transpose(conv, deconv_weights, output_shape=deconv_output, strides=deconv_stride, padding='SAME', data_format='NHWC')
    deconv = tf.nn.bias_add(deconv, deconv_biases, data_format='NHWC')

    return deconv

  def prelu(self, _x, i):
    """
    PreLU tensorflow implementation
    """
    alphas = tf.get_variable('alpha{}'.format(i), _x.get_shape()[-1], initializer=tf.constant_initializer(0.2), dtype=tf.float32)

    return tf.nn.relu(_x) - alphas * tf.nn.relu(-_x)

  def loss(self, Y, X):
    return tf.reduce_mean(tf.sqrt(tf.square(X - Y) + 1e-6)) + (1.0 - tf_ssim(Y, X)) * 0.5
