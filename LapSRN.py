import tensorflow as tf
import math
from utils import bilinear_upsample_weights, bicubic_downsample

class Model(object):

  def __init__(self, config):
    self.name = "LapSRN"
    self.model_params = [32, 2, 2]  #f, d, r in paper
    self.scale = config.scale
    self.radius = config.radius
    self.padding = config.padding
    self.images = config.images
    self.labels = config.labels
    self.batch = config.batch
    self.image_size = config.image_size - self.padding
    self.label_size = config.label_size
    self.c_dim = config.c_dim

  def model(self):
    d, m, r = self.model_params

    size = self.padding + 1
    conv = tf.contrib.layers.conv2d(self.images, d, size, 1, 'VALID', 'NHWC', activation_fn=None,
                             weights_initializer=tf.variance_scaling_initializer(scale=2.0), scope='feature_layer')

    input_height, input_width = tf.shape(self.labels)[1] // self.scale, tf.shape(self.labels)[2] // self.scale
    img, loss, reuse = tf.image.resize_image_with_crop_or_pad(self.images, input_height, input_width), 0.0, False
    for s in range(0, int(math.log(self.scale, 2))):
        with tf.variable_scope("recursive_block", reuse=reuse) as scope:
            features = conv
            for ri in range(r):
                for i in range(3, m+3):
                  conv = tf.nn.leaky_relu(conv)
                  conv = tf.contrib.layers.conv2d(conv, d, 3, 1, 'SAME', 'NHWC', activation_fn=None,
                                            weights_initializer=tf.variance_scaling_initializer(scale=2.0),
                                            scope='embedding_layer{}'.format(i-2))
                conv = tf.add(conv, features)
                scope.reuse_variables()

        with tf.variable_scope("upscaling", reuse=reuse):
            conv = tf.nn.leaky_relu(conv)
            conv = tf.contrib.layers.conv2d_transpose(conv, d, 4, 2, 'SAME', 'NHWC', activation_fn=None,
                                    weights_initializer=tf.variance_scaling_initializer(scale=2.0),
                                    scope='upsampling_layer')

            res = tf.contrib.layers.conv2d(conv, 1, 3, 1, 'SAME', 'NHWC', activation_fn=None,
                                    weights_initializer=tf.variance_scaling_initializer(scale=2.0),
                                    scope='res_layer')

        weights = bilinear_upsample_weights(2, 1)
        img = tf.contrib.layers.conv2d_transpose(img, 1, 4, 2, 'SAME', 'NHWC', activation_fn=None,
                                    weights_initializer=tf.constant_initializer(weights, dtype=tf.float32),
                                    biases_initializer=None, trainable=False) + res

        if s < int(math.log(self.scale, 2)) - 1:
            factor = self.scale // 2**(s+1)
            labels = bicubic_downsample(self.labels, factor)
        else:
            labels = self.labels
        loss = loss + tf.reduce_mean(tf.sqrt(tf.square(img - labels) + 1e-6))

        reuse = True

    self.loss_sum = loss
    return img

  def loss(self, Y, X):
    return self.loss_sum
