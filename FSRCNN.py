import tensorflow as tf
from utils import tf_ssim, bilinear_upsample_weights

class Model(object):

  def __init__(self, config):
    self.name = "FSRCNN"
    # Different model layer counts and filter sizes for FSRCNN vs FSRCNN-s (fast), (d, s, m) in paper
    model_params = [32, 0, 4, 1]
    self.GRL = True # global residual learning
    self.model_params = model_params
    self.scale = config.scale
    self.radius = config.radius
    self.padding = config.padding
    self.images = config.images
    self.batch = config.batch
    self.image_size = config.image_size - self.padding
    self.label_size = config.label_size
    self.c_dim = config.c_dim

  def model(self):

    d, s, m, r = self.model_params

    # Feature Extraction
    size = self.padding + 1
    weights = tf.get_variable('w1', shape=[size, size, 1, d], initializer=tf.variance_scaling_initializer())
    biases = tf.get_variable('b1', initializer=tf.zeros([d]))
    features = tf.nn.conv2d(self.images, weights, strides=[1,1,1,1], padding='VALID', data_format='NHWC')
    features = tf.nn.bias_add(features, biases, data_format='NHWC')

    # Shrinking
    if self.model_params[1] > 0:
      features = self.prelu(features, 1)
      weights = tf.get_variable('w2', shape=[1, 1, d, s], initializer=tf.variance_scaling_initializer())
      biases = tf.get_variable('b2', initializer=tf.zeros([s]))
      features = tf.nn.conv2d(features, weights, strides=[1,1,1,1], padding='SAME', data_format='NHWC')
      features = tf.nn.bias_add(features, biases, data_format='NHWC')
    else:
      s = d

    conv = features
    # Mapping (# mapping layers = m)
    with tf.variable_scope("mapping_block") as scope:
        for ri in range(r):
          for i in range(3, m + 3):
            weights = tf.get_variable('w{}'.format(i), shape=[3, 3, s, s], initializer=tf.variance_scaling_initializer())
            biases = tf.get_variable('b{}'.format(i), initializer=tf.zeros([s]))
            if i > 3:
              conv = self.prelu(conv, i)
            conv = tf.nn.conv2d(conv, weights, strides=[1,1,1,1], padding='SAME', data_format='NHWC')
            conv = tf.nn.bias_add(conv, biases, data_format='NHWC')
            if i == m + 2:
              conv = self.prelu(conv, m + 3)
              weights = tf.get_variable('w{}'.format(m + 3), shape=[1, 1, s, s], initializer=tf.variance_scaling_initializer())
              biases = tf.get_variable('b{}'.format(m + 3), initializer=tf.zeros([s]))
              conv = tf.nn.conv2d(conv, weights, strides=[1,1,1,1], padding='SAME', data_format='NHWC')
              conv = tf.nn.bias_add(conv, biases, data_format='NHWC')
              conv = tf.add(conv, features)
          scope.reuse_variables()
    conv = self.prelu(conv, 2)

    # Expanding
    if self.model_params[1] > 0:
      expand_weights = tf.get_variable('w{}'.format(m + 4), shape=[1, 1, s, d], initializer=tf.variance_scaling_initializer())
      expand_biases = tf.get_variable('b{}'.format(m + 4), initializer=tf.zeros([d]))
      conv = tf.nn.conv2d(conv, expand_weights, strides=[1,1,1,1], padding='SAME', data_format='NHWC')
      conv = tf.nn.bias_add(conv, expand_biases, data_format='NHWC')
      conv = self.prelu(conv, m + 4)

    # Deconvolution
    deconv_size = self.radius * self.scale * 2 + 1
    deconv_weights = tf.get_variable('w{}'.format(m + 5), shape=[deconv_size, deconv_size, 1, d], initializer=tf.variance_scaling_initializer(scale=0.01))
    deconv_biases = tf.get_variable('b{}'.format(m + 5), initializer=tf.zeros([1]))
    deconv_output = [self.batch, self.label_size, self.label_size, self.c_dim]
    deconv_stride = [1,  self.scale, self.scale, 1]
    deconv = tf.nn.conv2d_transpose(conv, deconv_weights, output_shape=deconv_output, strides=deconv_stride, padding='SAME', data_format='NHWC')
    deconv = tf.nn.bias_add(deconv, deconv_biases, data_format='NHWC')

    if self.GRL:
        # Deconvolution 2
        upsample_filter = bilinear_upsample_weights(self.scale, self.c_dim)
        self.biases['b{}'.format(m + 6)] = tf.get_variable('b{}'.format(m + 6), initializer=tf.constant(1, shape=[1]))
        deconv_output = [self.batch, self.label_size, self.label_size, self.c_dim]
        deconv_stride = [1, self.scale, self.scale, 1]
        img = tf.image.resize_image_with_crop_or_pad(self.images, self.image_size, self.image_size)
        deconv += tf.nn.conv2d_transpose(img, upsample_filter, output_shape=deconv_output, strides=deconv_stride, padding='SAME')

    return deconv

  def prelu(self, _x, i):
    """
    PreLU tensorflow implementation
    """
    alphas = tf.get_variable('alpha{}'.format(i), _x.get_shape()[-1], initializer=tf.constant_initializer(0.2), dtype=tf.float32)

    return tf.nn.relu(_x) - alphas * tf.nn.relu(-_x)

  def loss(self, Y, X):
    return tf.reduce_mean(tf.sqrt(tf.square(X - Y) + 1e-6)) + (1.0 - tf_ssim(Y, X)) * 0.5
