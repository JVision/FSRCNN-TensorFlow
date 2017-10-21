"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py
from math import ceil
import struct
import io
from random import randrange

import tensorflow as tf
from PIL import Image  
import numpy as np
from multiprocessing import Pool, Lock, active_children

import pdb

FLAGS = tf.app.flags.FLAGS

def read_data(path):
  """
  Read h5 format data file
  
  Args:
    path: file path of desired file

  Returns:
    data: '.h5' file format that contains train data values
    label: '.h5' file format that contains train label values
  """
  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    return data, label

def preprocess(path, scale=3, distort=False):
  """
  Preprocess single image file 
    (1) Read original image as YCbCr format
    (2) Normalize
    (3) Downsampled by scale factor
  """

  image = Image.open(path).convert('L')
  (width, height) = image.size
  label_ = np.fromstring(image.tobytes(), dtype=np.uint8).reshape((height, width)) / 255
  image.close()

  cropped_image = Image.fromarray(modcrop(label_, scale))

  (width, height) = cropped_image.size
  new_width, new_height = int(width / scale), int(height / scale)
  scaled_image = cropped_image.resize((new_width, new_height), Image.BICUBIC)
  cropped_image.close()

  (width, height) = scaled_image.size
  input_ = np.array(scaled_image.getdata()).astype(np.float).reshape((height, width))

  if randrange(3) == 2 and distort==True:
      buf = io.BytesIO()
      i = Image.fromarray(input_ * 255)
      i.convert('RGB').save(buf, "JPEG", quality=randrange(50, 99, 5))
      buf.seek(0)
      scaled_image = Image.open(buf).convert('L')
      input_ = np.fromstring(scaled_image.tobytes(), dtype=np.uint8).reshape((height, width)) / 255

  return input_, label_

def prepare_data(sess, dataset):
  """
  Args:
    dataset: choose train dataset or test dataset
    
    For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
  """
  if FLAGS.train:
    filenames = os.listdir(dataset)
    data_dir = os.path.join(os.getcwd(), dataset)
  else:
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "Set5")
  data = sorted(glob.glob(os.path.join(data_dir, "*.bmp")))

  return data

def make_data(sess, checkpoint_dir, data, label):
  """
  Make input data as h5 file format
  Depending on 'train' (flag value), savepath would be changed.
  """
  if FLAGS.train:
    savepath = os.path.join(os.getcwd(), '{}/train.h5'.format(checkpoint_dir))
  else:
    savepath = os.path.join(os.getcwd(), '{}/test.h5'.format(checkpoint_dir))

  with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=data)
    hf.create_dataset('label', data=label)

def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def train_input_worker(args):
  image_data, config = args
  image_size, label_size, stride, scale, padding, distort = config

  single_input_sequence, single_label_sequence = [], []

  input_, label_ = preprocess(image_data, scale, distort=distort)

  if len(input_.shape) == 3:
    h, w, _ = input_.shape
  else:
    h, w = input_.shape

  for x in range(0, h - image_size + 1, stride):
    for y in range(0, w - image_size + 1, stride):
      sub_input = input_[x : x + image_size, y : y + image_size]
      x_loc, y_loc = x + padding, y + padding
      sub_label = label_[x_loc * scale : x_loc * scale + label_size, y_loc * scale : y_loc * scale + label_size]

      sub_input = sub_input.reshape([image_size, image_size, 1])
      sub_label = sub_label.reshape([label_size, label_size, 1])
      
      single_input_sequence.append(sub_input)
      single_label_sequence.append(sub_label)

  return [single_input_sequence, single_label_sequence]


def thread_train_setup(config):
  """
  Spawns |config.threads| worker processes to pre-process the data

  This has not been extensively tested so use at your own risk.
  Also this is technically multiprocessing not threading, I just say thread
  because it's shorter to type.
  """
  sess = config.sess

  # Load data path
  data = prepare_data(sess, dataset=config.data_dir)

  # Initialize multiprocessing pool with # of processes = config.threads
  pool = Pool(config.threads)

  # Distribute |images_per_thread| images across each worker process
  config_values = [config.image_size, config.label_size, config.stride, config.scale, config.padding // 2, config.distort]
  images_per_thread = len(data) // config.threads
  workers = []
  for thread in range(config.threads):
    args_list = [(data[i], config_values) for i in range(thread * images_per_thread, (thread + 1) * images_per_thread)]
    worker = pool.map_async(train_input_worker, args_list)
    workers.append(worker)
  print("{} worker processes created".format(config.threads))

  pool.close()

  results = []
  for i in range(len(workers)):
    print("Waiting for worker process {}".format(i))
    results.extend(workers[i].get(timeout=240))
    print("Worker process {} done".format(i))

  print("All worker processes done!")

  sub_input_sequence, sub_label_sequence = [], []

  for image in range(len(results)):
    single_input_sequence, single_label_sequence = results[image]
    sub_input_sequence.extend(single_input_sequence)
    sub_label_sequence.extend(single_label_sequence)

  arrdata = np.asarray(sub_input_sequence)
  arrlabel = np.asarray(sub_label_sequence)

  make_data(sess, config.checkpoint_dir, arrdata, arrlabel)


def train_input_setup(config):
  """
  Read image files, make their sub-images, and save them as a h5 file format.
  """
  sess = config.sess
  image_size, label_size, stride, scale, padding = config.image_size, config.label_size, config.stride, config.scale, config.padding // 2

  # Load data path
  data = prepare_data(sess, dataset=config.data_dir)

  sub_input_sequence, sub_label_sequence = [], []

  for i in range(len(data)):
    input_, label_ = preprocess(data[i], scale, distort=config.distort)

    if len(input_.shape) == 3:
      h, w, _ = input_.shape
    else:
      h, w = input_.shape

    for x in range(0, h - image_size + 1, stride):
      for y in range(0, w - image_size + 1, stride):
        sub_input = input_[x : x + image_size, y : y + image_size]
        x_loc, y_loc = x + padding, y + padding
        sub_label = label_[x_loc * scale : x_loc * scale + label_size, y_loc * scale : y_loc * scale + label_size]

        sub_input = sub_input.reshape([image_size, image_size, 1])
        sub_label = sub_label.reshape([label_size, label_size, 1])
        
        sub_input_sequence.append(sub_input)
        sub_label_sequence.append(sub_label)

  arrdata = np.asarray(sub_input_sequence)
  arrlabel = np.asarray(sub_label_sequence)

  make_data(sess, config.checkpoint_dir, arrdata, arrlabel)


def test_input_setup(config):
  """
  Read image files, make their sub-images, and save them as a h5 file format.
  """
  sess = config.sess
  image_size, label_size, stride, scale, padding = config.image_size, config.label_size, config.stride, config.scale, config.padding // 2

  # Load data path
  data = prepare_data(sess, dataset="Test")

  sub_input_sequence, sub_label_sequence = [], []

  pic_index = 2 # Index of image based on lexicographic order in data folder
  input_, label_ = preprocess(data[pic_index], config.scale)

  if len(input_.shape) == 3:
    h, w, _ = input_.shape
  else:
    h, w = input_.shape

  nx, ny = 0, 0
  for x in range(0, h - image_size + 1, stride):
    nx += 1
    ny = 0
    for y in range(0, w - image_size + 1, stride):
      ny += 1
      sub_input = input_[x : x + image_size, y : y + image_size]
      x_loc, y_loc = x + padding, y + padding
      sub_label = label_[x_loc * scale : x_loc * scale + label_size, y_loc * scale : y_loc * scale + label_size]

      sub_input = sub_input.reshape([image_size, image_size, 1])
      sub_label = sub_label.reshape([label_size, label_size, 1])

      sub_input_sequence.append(sub_input)
      sub_label_sequence.append(sub_label)

  arrdata = np.asarray(sub_input_sequence)
  arrlabel = np.asarray(sub_label_sequence)

  make_data(sess, config.checkpoint_dir, arrdata, arrlabel)

  return nx, ny

def save_params(sess, params):
  param_dir = "params/"

  if not os.path.exists(param_dir):
    os.makedirs(param_dir)

  h = open(param_dir + "weights{}.txt".format('_'.join(str(i) for i in params)), 'w')

  variables = dict((var.name, sess.run(var)) for var in tf.trainable_variables())

  for name, weights in variables.items():
    h.write("{} =\n".format(name[:name.index(':')]))

    if len(weights.shape) < 4:
        h.write("{}\n\n".format(weights.flatten().tolist()))
    else:
        h.write("[")
        sep = False
        for filter_x in range(len(weights)):
          for filter_y in range(len(weights[filter_x])):
            filter_weights = weights[filter_x][filter_y]
            for input_channel in range(len(filter_weights)):
              for output_channel in range(len(filter_weights[input_channel])):
                val = filter_weights[input_channel][output_channel]
                if sep:
                    h.write(', ')
                h.write("{}".format(val))
                sep = True
              h.write("\n  ")
        h.write("]\n\n")

  h.close()

def merge(images, size):
  """
  Merges sub-images back into original image size
  """
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1], size[2]))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image

  return img

def array_image_save(array, image_path):
  """
  Converts np array to image and saves it
  """
  image = Image.fromarray(array)
  if image.mode != 'RGB':
    image = image.convert('RGB')
  image.save(image_path)
  print("Saved image: {}".format(image_path))

def _tf_fspecial_gauss(sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    size = int(sigma * 3) * 2 + 1

    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, l=False, mean_metric=True, sigma=1.7):
    window = _tf_fspecial_gauss(sigma)
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID', data_format='NHWC')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1], padding='VALID', data_format='NHWC')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1], padding='VALID', data_format='NHWC') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1], padding='VALID', data_format='NHWC') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1], padding='VALID', data_format='NHWC') - mu1_mu2
    if cs_map:
        value = ((2.0*mu1_mu2 + C1)/(mu1_sq + mu2_sq + C1) if l==True else 1.0,
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_ms_ssim(img1, img2, level=3, sigma=1.5):
    weight = tf.constant([[1.0], [0.5, 0.5], [0.3, 0.4, 0.3], None, [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]][level-1], dtype=tf.float32)
    window = _tf_fspecial_gauss(0.5)
    ml = []
    mcs = []
    for i in range(level):
        l_map, cs_map = tf_ssim(img1, img2, cs_map=True, l=(i==level-1), mean_metric=False, sigma=sigma)
        ml.append(tf.reduce_mean(l_map))
        mcs.append(tf.reduce_mean(cs_map))
        #img1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='SAME')
        #img2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1], padding='SAME')
        size = img1.shape[1].value // 2
        img1 = tf.image.resize_bicubic(img1, [size, size])
        img2 = tf.image.resize_bicubic(img2, [size, size])

    # list to tensor of dim D+1
    ml = tf.stack(ml, axis=0)
    mcs = tf.stack(mcs, axis=0)

    value = tf.reduce_prod(tf.pow(mcs, weight)) * tf.pow(ml[level-1], weight[level-1])

    return value

def bilinear_upsample_weights(factor, channels):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """
    filter_size = 2 * factor
    center = factor - (1 if factor % 2 == 1 else 0.5)
    og = np.ogrid[:filter_size, :filter_size]
    upsample_kernel = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weights = np.zeros((filter_size, filter_size, channels, channels), dtype=np.float32)
    for i in range(channels):
        weights[:, :, i, i] = upsample_kernel
    return weights

def bicubic_kernel(x, a=-0.5):
    """https://clouard.users.greyc.fr/Pantheon/experiments/rescaling/index-en.html#bicubic"""
    if abs(x) <= 1:
      return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
    elif 1 < abs(x) and abs(x) < 2:
      return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
    else:
      return 0

def build_filter(factor, channels=1):
    size = factor * 4
    k = np.zeros((size), dtype=np.float32)
    for i in range(size):
        x = (1 / factor) * (i - np.floor(size / 2) + 0.5)
        k[i] = bicubic_kernel(x)
    k = k / np.sum(k)
    k = np.outer(k, k)
    weights = np.zeros((size, size, channels, channels), dtype=np.float32)
    for i in range(channels):
        weights[:, :, i, i] = k
    return weights

def bicubic_downsample(x, factor):
    """Downsample x by a factor of factor, using the filter built by build_filter()
    x: a rank 4 tensor with format NHWC
    factor: downsampling factor (ex: factor=2 means the output size is (h/2, w/2))
    """
    # using padding calculations from https://www.tensorflow.org/api_guides/python/nn#Convolution
    kernel_size = factor * 4
    padding = kernel_size - factor
    pad_top = padding // 2
    pad_bottom = padding - pad_top
    pad_left = padding // 2
    pad_right = padding - pad_left
    # apply mirror padding
    x = tf.pad(x, [[0,0], [pad_top,pad_bottom], [pad_left,pad_right], [0,0]], mode='REFLECT')
    # downsampling performed by strided conv
    x = tf.contrib.layers.conv2d(x, 1, kernel_size, factor, 'VALID', 'NHWC', activation_fn=None, trainable=False,
        weights_initializer=tf.constant_initializer(build_filter(factor=factor), dtype=tf.float32), biases_initializer=None)
    return x
