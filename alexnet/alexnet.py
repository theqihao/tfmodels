#%%
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from datetime import datetime
import math
import time
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
#tf.logging.set_verbosity(tf.logging.DEBUG)



def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def inference(images):
    parameters = []
    # conv1
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]


  # pool1
    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1')
    print_activations(pool1)

  # conv2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv2)

  # pool2
    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool2')
    print_activations(pool2)

  # conv3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)

  # conv4
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv4)

  # conv5
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)

  # pool5
    pool5 = tf.nn.max_pool(conv5,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool5')
    print_activations(pool5)
    # pool5   [32, 6, 6, 256]
    with tf.name_scope("fc6") as scope:
        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
        dropout6 = dropout(fc6, 0.5)
    with tf.name_scope("fc7") as scope:
        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, 0.5)
    with tf.name_scope("fc8") as scope:
        # 8th Layer: FC and return unscaled activations
        fc8 = fc(dropout7, 4096, 1000, relu=False, name='fc8')
    print(fc8)
    return fc8, parameters

def fc(x, num_in, num_out, name, relu=True):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)

def time_tensorflow_run(session, target, info_string):
#  """Run the computation to obtain the target tensor and print timing stats.
#
#  Args:
#    session: the TensorFlow session to run the computation under.
#    target: the target Tensor that is passed to the session's run() function.
#    info_string: a string summarizing this run, to be printed with the stats.
#
#  Returns:
#    None
#  """
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print ('%s: step %d, duration = %.3f' %
                       (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
           (datetime.now(), info_string, num_batches, mn, sd))



def run_benchmark():
#  """Run the benchmark on AlexNet."""
    with tf.Graph().as_default():
    # Generate some dummy images.
        image_size = 224
    # Note that our padding definition is slightly different the cuda-convnet.
    # In order to force the model to start with the same activations sizes,
    # we add 3 to the image_size and employ VALID padding above.
        images = tf.Variable(tf.random_normal([batch_size,
                                           image_size,
                                           image_size, 3],
                                          dtype=tf.float32,
                                          stddev=1e-1))
        labels = tf.Variable(tf.random_normal([batch_size, 1000],
                                               dtype=tf.float32,
                                               stddev=1e-1))
    # Build a Graph that computes the logits predictions from the
    # inference model.
        fc8, parameters = inference(images)

    # Build an initialization operation.
        init = tf.global_variables_initializer()

    # Start running operations on the Graph.
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True      #????????  
        config.gpu_options.allocator_type = 'BFC'
        sess = tf.Session(config=config)
        sess.run(init)
        """
        sess = tf.Session()
        sess.run(init)
        """
    # Run the forward benchmark.
        #time_tensorflow_run(sess, pool5, "Forward")

    # Add a simple objective so we can calculate the backward pass.
        #objective = tf.nn.l2_loss(pool5)
    # Compute the gradient with respect to all the parameters.
        #grad = tf.gradients(objective, parameters)
    # Run the backward benchmark.
        run_metadata = tf.RunMetadata()
        mygrad = tf.train.GradientDescentOptimizer(0.1).minimize(tf.nn.l2_loss(fc8-labels))
        for i in range(num_batches):
          _ = sess.run(mygrad, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
        # time_tensorflow_run(sess, grad, "Forward-backward")
        # tf.contrib.tfprof.model_analyzer.print_model_analysis(
        #     tf.get_default_graph(),
        #     run_meta=run_metadata,
        #     tfprof_options=tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY)
        # tf.profiler.profile(
        #   tf.get_default_graph(),
        #   run_meta=run_metadata,
        #   cmd='op',
        #   options=tf.profiler.ProfileOptionBuilder.time_and_memory())

        # tf.profiler.profile(
        #   tf.get_default_graph(),
        #   run_meta=run_metadata,
        #   cmd='code',
        #   options=tf.profiler.ProfileOptionBuilder.time_and_memory())

        # tf.profiler.profile(
        #   tf.get_default_graph(),
        #   run_meta=run_metadata,
        #   cmd='scope',
        #   options=tf.profiler.ProfileOptionBuilder.time_and_memory())
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
        tf.profiler.profile(
          tf.get_default_graph(),
          run_meta=run_metadata,
          cmd='op',
          options=opts)
        # output_dir='/home/liubo/tensorflow/tfmodels/alexnet/'
        # with tf.gfile.Open(os.path.join(output_dir, "run_meta"), "w") as f:
        #   f.write(run_metadata.SerializeToString())
        # tf.train.write_graph(sess.graph, './', 'train.pbtxt')
        #tf.profiler.write_op_log(graph, log_dir, op_log=None)
batch_size=500
num_batches=100
run_benchmark()

