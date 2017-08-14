
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np



slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def inception_v1_base(inputs,
                      final_endpoint='Mixed_5c',
                      scope='InceptionV1'):
  """Defines the Inception V1 base architecture.
  This architecture is defined in:
    Going deeper with convolutions
    Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
    http://arxiv.org/pdf/1409.4842v1.pdf.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
      'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c',
      'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e',
      'Mixed_4f', 'MaxPool_5a_2x2', 'Mixed_5b', 'Mixed_5c']
    scope: Optional variable_scope.
  Returns:
    A dictionary from components of the network to the corresponding activation.
  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values.
  """
  end_points = {}
  with tf.variable_scope(scope, 'InceptionV1', [inputs]):
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_initializer=trunc_normal(0.01)):
      with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                          stride=1, padding='SAME'):
        end_point = 'Conv2d_1a_7x7'
        net = slim.conv2d(inputs, 64, [7, 7], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
        end_point = 'MaxPool_2a_3x3'
        net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
        end_point = 'Conv2d_2b_1x1'
        net = slim.conv2d(net, 64, [1, 1], scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
        end_point = 'Conv2d_2c_3x3'
        net = slim.conv2d(net, 192, [3, 3], scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
        end_point = 'MaxPool_3a_3x3'
        net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_3b'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 128, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 32, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_3c'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 192, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'MaxPool_4a_3x3'
        net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4b'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 208, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 48, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4c'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4d'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 256, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4e'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 144, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 288, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4f'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'MaxPool_5a_2x2'
        net = slim.max_pool2d(net, [2, 2], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_5b'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0a_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_5c'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 384, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def inception_v1(inputs,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV1'):
  """Defines the Inception V1 architecture.
  This architecture is defined in:
    Going deeper with convolutions
    Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
    http://arxiv.org/pdf/1409.4842v1.pdf.
  The default image size used to train this network is 224x224.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    dropout_keep_prob: the percentage of activation values that are retained.
    prediction_fn: a function to get predictions out of logits.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is of
        shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, num_classes]
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """
  # Final pooling and prediction
  with tf.variable_scope(scope, 'InceptionV1', [inputs, num_classes],
                         reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = inception_v1_base(inputs, scope=scope)
      with tf.variable_scope('Logits'):
        net = slim.avg_pool2d(net, [7, 7], stride=1, scope='AvgPool_0a_7x7')
        net = slim.dropout(net,
                           dropout_keep_prob, scope='Dropout_0b')
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_0c_1x1')
        if spatial_squeeze:
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

        end_points['Logits'] = logits
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points


inception_v1.default_image_size = 224

slim = tf.contrib.slim

def inception_arg_scope(weight_decay=0.00004,
                        use_batch_norm=True,
                        batch_norm_decay=0.9997,
                        batch_norm_epsilon=0.001):
  """Defines the default arg scope for inception models.
  Args:
    weight_decay: The weight decay to use for regularizing the model.
    use_batch_norm: "If `True`, batch_norm is applied after each convolution.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.
  Returns:
    An `arg_scope` to use for the inception models.
  """
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      # collection containing update_ops.
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }
  if use_batch_norm:
    normalizer_fn = slim.batch_norm
    normalizer_params = batch_norm_params
  else:
    normalizer_fn = None
    normalizer_params = {}
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=slim.variance_scaling_initializer(),
        activation_fn=tf.nn.relu,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params) as sc:
      return sc

inception_v1_arg_scope = inception_arg_scope




class InceptionV1Test(tf.test.TestCase):

  def testBuildClassificationNetwork(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000

    inputs = tf.random_uniform((batch_size, height, width, 3))
    logits, end_points = inception_v1(inputs, num_classes)
    self.assertTrue(logits.op.name.startswith('InceptionV1/Logits'))
    self.assertListEqual(logits.get_shape().as_list(),
                         [batch_size, num_classes])
    self.assertTrue('Predictions' in end_points)
    self.assertListEqual(end_points['Predictions'].get_shape().as_list(),
                         [batch_size, num_classes])

  def testBuildBaseNetwork(self):
    batch_size = 5
    height, width = 224, 224

    inputs = tf.random_uniform((batch_size, height, width, 3))
    mixed_6c, end_points = inception_v1_base(inputs)
    self.assertTrue(mixed_6c.op.name.startswith('InceptionV1/Mixed_5c'))
    self.assertListEqual(mixed_6c.get_shape().as_list(),
                         [batch_size, 7, 7, 1024])
    expected_endpoints = ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
                          'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b',
                          'Mixed_3c', 'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c',
                          'Mixed_4d', 'Mixed_4e', 'Mixed_4f', 'MaxPool_5a_2x2',
                          'Mixed_5b', 'Mixed_5c']
    self.assertItemsEqual(end_points.keys(), expected_endpoints)

  def testBuildOnlyUptoFinalEndpoint(self):
    batch_size = 5
    height, width = 224, 224
    endpoints = ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
                 'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c',
                 'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d',
                 'Mixed_4e', 'Mixed_4f', 'MaxPool_5a_2x2', 'Mixed_5b',
                 'Mixed_5c']
    for index, endpoint in enumerate(endpoints):
      with tf.Graph().as_default():
        inputs = tf.random_uniform((batch_size, height, width, 3))
        out_tensor, end_points = inception_v1_base(
            inputs, final_endpoint=endpoint)
        self.assertTrue(out_tensor.op.name.startswith(
            'InceptionV1/' + endpoint))
        self.assertItemsEqual(endpoints[:index+1], end_points)

  def testBuildAndCheckAllEndPointsUptoMixed5c(self):
    batch_size = 5
    height, width = 224, 224

    inputs = tf.random_uniform((batch_size, height, width, 3))
    _, end_points = inception_v1_base(inputs,
                                                final_endpoint='Mixed_5c')
    endpoints_shapes = {'Conv2d_1a_7x7': [5, 112, 112, 64],
                        'MaxPool_2a_3x3': [5, 56, 56, 64],
                        'Conv2d_2b_1x1': [5, 56, 56, 64],
                        'Conv2d_2c_3x3': [5, 56, 56, 192],
                        'MaxPool_3a_3x3': [5, 28, 28, 192],
                        'Mixed_3b': [5, 28, 28, 256],
                        'Mixed_3c': [5, 28, 28, 480],
                        'MaxPool_4a_3x3': [5, 14, 14, 480],
                        'Mixed_4b': [5, 14, 14, 512],
                        'Mixed_4c': [5, 14, 14, 512],
                        'Mixed_4d': [5, 14, 14, 512],
                        'Mixed_4e': [5, 14, 14, 528],
                        'Mixed_4f': [5, 14, 14, 832],
                        'MaxPool_5a_2x2': [5, 7, 7, 832],
                        'Mixed_5b': [5, 7, 7, 832],
                        'Mixed_5c': [5, 7, 7, 1024]}

    self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
    for endpoint_name in endpoints_shapes:
      expected_shape = endpoints_shapes[endpoint_name]
      self.assertTrue(endpoint_name in end_points)
      self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                           expected_shape)

  def testModelHasExpectedNumberOfParameters(self):
    batch_size = 5
    height, width = 224, 224
    inputs = tf.random_uniform((batch_size, height, width, 3))
    with slim.arg_scope(inception_v1_arg_scope()):
      inception_v1_base(inputs)
    total_params, _ = slim.model_analyzer.analyze_vars(
        slim.get_model_variables())
    self.assertAlmostEqual(5607184, total_params)

  def testHalfSizeImages(self):
    batch_size = 5
    height, width = 112, 112

    inputs = tf.random_uniform((batch_size, height, width, 3))
    mixed_5c, _ = inception_v1_base(inputs)
    self.assertTrue(mixed_5c.op.name.startswith('InceptionV1/Mixed_5c'))
    self.assertListEqual(mixed_5c.get_shape().as_list(),
                         [batch_size, 4, 4, 1024])

  def testUnknownImageShape(self):
    tf.reset_default_graph()
    batch_size = 2
    height, width = 224, 224
    num_classes = 1000
    input_np = np.random.uniform(0, 1, (batch_size, height, width, 3))
    with self.test_session() as sess:
      inputs = tf.placeholder(tf.float32, shape=(batch_size, None, None, 3))
      logits, end_points = inception_v1(inputs, num_classes)
      self.assertTrue(logits.op.name.startswith('InceptionV1/Logits'))
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])
      pre_pool = end_points['Mixed_5c']
      feed_dict = {inputs: input_np}
      tf.global_variables_initializer().run()
      pre_pool_out = sess.run(pre_pool, feed_dict=feed_dict)
      self.assertListEqual(list(pre_pool_out.shape), [batch_size, 7, 7, 1024])

  def testUnknowBatchSize(self):
    batch_size = 32
    height, width = 224, 224
    num_classes = 1000

    inputs = tf.placeholder(tf.float32, (None, height, width, 3))
    logits, _ = inception_v1(inputs, num_classes)
    self.assertTrue(logits.op.name.startswith('InceptionV1/Logits'))
    self.assertListEqual(logits.get_shape().as_list(),
                         [None, num_classes])
    images = tf.random_uniform((batch_size, height, width, 3))

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(logits, {inputs: images.eval()})
      self.assertEquals(output.shape, (batch_size, num_classes))

  def testEvaluation(self):
    batch_size = 2
    height, width = 224, 224
    num_classes = 1000

    eval_inputs = tf.random_uniform((batch_size, height, width, 3))
    logits, _ = inception_v1(eval_inputs, num_classes,
                                       is_training=False)
    predictions = tf.argmax(logits, 1)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(predictions)
      self.assertEquals(output.shape, (batch_size,))

  def testTrainEvalWithReuse(self):
    train_batch_size = 5
    eval_batch_size = 2
    height, width = 224, 224
    num_classes = 1000

    train_inputs = tf.random_uniform((train_batch_size, height, width, 3))
    inception_v1(train_inputs, num_classes)
    eval_inputs = tf.random_uniform((eval_batch_size, height, width, 3))
    logits, _ = inception_v1(eval_inputs, num_classes, reuse=True)
    predictions = tf.argmax(logits, 1)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(predictions)
      self.assertEquals(output.shape, (eval_batch_size,))

  def testLogitsNotSqueezed(self):
    num_classes = 25
    images = tf.random_uniform([1, 224, 224, 3])
    logits, _ = inception_v1(images,
                                       num_classes=num_classes,
                                       spatial_squeeze=False)

    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      logits_out = sess.run(logits)
      self.assertListEqual(list(logits_out.shape), [1, 1, 1, num_classes])




def run_benchmark():
  """Run the benchmark on AlexNet."""
  with tf.Graph().as_default():
    """
    # Generate some dummy images.
    batch_size = 32
    #num_batches = 100
    image_size = 224
    inputs = tf.Variable(tf.random_normal([batch_size,
                                           image_size,
                                           image_size, 3],
                                          dtype=tf.float32,
                                          stddev=1e-1))

    with slim.arg_scope(incepton_v1_arg_scope()):
      logits, end_points = inception_v1(inputs, is_training = False)                                      

    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=config)
    sess.run(init)
    
    #time_tensorflow_run(sess, logits, "Forward")

    objective = tf.nn.l2_loss(logits)
    grad = tf.gradients(objective, end_points)
    time_tensorflow_run(sess, grad, "Forward-backward")
    """
    #对Inception V1进行运算性能测试 
    batch_size = 16
    height, width = 224, 224 
    num_classes = 1000
    num_batches = 10
    inputs = tf.random_uniform((batch_size, height, width, 3)) 
    outputs = tf.random_uniform((batch_size, num_classes)) 
    #with slim.arg_scope(inception_arg_scope()): 
    logits, end_points = inception_v1(inputs, num_classes) 
    #logits, end_points = inception_v1(inputs, 480, is_training = False) 
    init = tf.global_variables_initializer() 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  
    config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=config)
    sess.run(init)
    print(inputs)
    print(outputs)
    #time_tensorflow_run(sess, logits, "Forward")  
    #objective = tf.nn.l2_loss(logits)
    #sgrad = tf.gradients(objective, end_points['Logits'])

    #print(logits)
    #print(end_points['Logits'])
    #objective = tf.nn.l2_loss(logits)
    #grad = tf.gradients(logits, inputs)
    #AdamOptimizer GradientDescentOptimizer tf.nn.l2_loss
    #mygrad = tf.train.GradientDescentOptimizer(0.1).minimize(tf.abs(logits-outputs))
    mygrad = tf.train.GradientDescentOptimizer(0.1).minimize(tf.nn.l2_loss(logits-outputs))
    run_metadata = tf.RunMetadata()
    for i in range(num_batches):  
      _ = sess.run(mygrad, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
    opts = (tf.profiler.ProfileOptionBuilder()
          .with_max_depth(1000)
          .select(['bytes','peak_bytes','residual_bytes','output_bytes'])
          .account_displayed_op_only(False)
          .with_stdout_output()
          .with_min_memory(1, 1, 1, 1)
          .build())
    tf.profiler.profile(
          tf.get_default_graph(),
          run_meta=run_metadata,
          cmd='scope',
          options=opts)
if __name__ == '__main__':
  run_benchmark()
  #tf.test.main()
