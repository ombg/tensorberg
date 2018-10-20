import tensorflow as tf

#TODO Only because of `shape = int(np.prod(self.pool5.get_shape()[1:]))`
import numpy as np

class Vgg16:
    def __init__(self, config, data_loader=None):


        self.config = config

        self.images = None
        self.labels = None
        self.out_argmax = None
        self.loss = None
        self.accuracy = None
        self.optimizer = None
        self.train_step = None
        self.softmax = None
        self.parameters = []

        if data_loader != None:
            self.images, self.labels = data_loader.get_input()
        else:
            self.images = tf.placeholder(tf.float32, [None, 224, 224, 3])
            self.labels = tf.placeholder(tf.int32, [None])

        self.build_model()

    def load_weights_from_numpy(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print( i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))

    def build_model(self):

        """
        Inputs to the network
        """
        tf.add_to_collection('inputs', self.images)
        tf.add_to_collection('inputs', self.labels)

        """
        Network Architecture
        """
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images_avrgd = self.images - mean

        # conv1_1
        with tf.variable_scope('conv1_1') as scope:
            kernel = tf.get_variable(name='weights',
                                     shape=[3, 3, 3, 64],
                                     initializer=tf.glorot_uniform_initializer(),
                                     regularizer=tf.nn.l2_loss,
                                     trainable=True)
            biases = tf.get_variable(name='biases',
                                     shape=[64],
                                     initializer=tf.zeros_initializer(),
                                     trainable=True)
            # conv is tf.Tensor. It is the output volume of shape [batchsize, 224,224,64]
            # Cool: This info is available before the session starts existing.
            conv = tf.nn.conv2d(images_avrgd, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            # Store final result of conv1_1 in an instance variable
            self.conv1_1 = tf.nn.relu(out, name='relu')
            # Add this layer's parameters to the instance's list of weights.
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.variable_scope('conv1_2') as scope:
            kernel = tf.get_variable(name='weights',
                                     shape=[3, 3, 64, 64],
                                     initializer=tf.glorot_uniform_initializer(),
                                     regularizer=tf.nn.l2_loss,
                                     trainable=True)
            biases = tf.get_variable(name='biases',
                                     shape=[64],
                                     initializer=tf.zeros_initializer(),
                                     trainable=True)
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name='relu')
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.variable_scope('conv2_1') as scope:
            kernel = tf.get_variable(name='weights',
                                     shape=[3, 3, 64, 128],
                                     initializer=tf.glorot_uniform_initializer(),
                                     regularizer=tf.nn.l2_loss,
                                     trainable=True)
            biases = tf.get_variable(name='biases',
                                     shape=[128],
                                     initializer=tf.zeros_initializer(),
                                     trainable=True)
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name='relu')
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.variable_scope('conv2_2') as scope:
            kernel = tf.get_variable(name='weights',
                                     shape=[3, 3, 128, 128],
                                     initializer=tf.glorot_uniform_initializer(),
                                     regularizer=tf.nn.l2_loss,
                                     trainable=True)
            biases = tf.get_variable(name='biases',
                                     shape=[128],
                                     initializer=tf.zeros_initializer(),
                                     trainable=True)
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name='relu')
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.variable_scope('conv3_1') as scope:
            kernel = tf.get_variable(name='weights',
                                     shape=[3, 3, 128, 256],
                                     initializer=tf.glorot_uniform_initializer(),
                                     regularizer=tf.nn.l2_loss,
                                     trainable=True)
            biases = tf.get_variable(name='biases',
                                     shape=[256],
                                     initializer=tf.zeros_initializer(),
                                     trainable=True)
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name='relu')
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.variable_scope('conv3_2') as scope:
            kernel = tf.get_variable(name='weights',
                                     shape=[3, 3, 256, 256],
                                     initializer=tf.glorot_uniform_initializer(),
                                     regularizer=tf.nn.l2_loss,
                                     trainable=True)
            biases = tf.get_variable(name='biases',
                                     shape=[256],
                                     initializer=tf.zeros_initializer(),
                                     trainable=True)
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name='relu')
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.variable_scope('conv3_3') as scope:
            kernel = tf.get_variable(name='weights',
                                     shape=[3, 3, 256, 256],
                                     initializer=tf.glorot_uniform_initializer(),
                                     regularizer=tf.nn.l2_loss,
                                     trainable=True)
            biases = tf.get_variable(name='biases',
                                     shape=[256],
                                     initializer=tf.zeros_initializer(),
                                     trainable=True)
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name='relu')
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.variable_scope('conv4_1') as scope:
            kernel = tf.get_variable(name='weights',
                                     shape=[3, 3, 256, 512],
                                     initializer=tf.glorot_uniform_initializer(),
                                     regularizer=tf.nn.l2_loss,
                                     trainable=True)
            biases = tf.get_variable(name='biases',
                                     shape=[512],
                                     initializer=tf.zeros_initializer(),
                                     trainable=True)
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name='relu')
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.variable_scope('conv4_2') as scope:
            kernel = tf.get_variable(name='weights',
                                     shape=[3, 3, 512, 512],
                                     initializer=tf.glorot_uniform_initializer(),
                                     regularizer=tf.nn.l2_loss,
                                     trainable=True)
            biases = tf.get_variable(name='biases',
                                     shape=[512],
                                     initializer=tf.zeros_initializer(),
                                     trainable=True)
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name='relu')
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.variable_scope('conv4_3') as scope:
            kernel = tf.get_variable(name='weights',
                                     shape=[3, 3, 512, 512],
                                     initializer=tf.glorot_uniform_initializer(),
                                     regularizer=tf.nn.l2_loss,
                                     trainable=True)
            biases = tf.get_variable(name='biases',
                                     shape=[512],
                                     initializer=tf.zeros_initializer(),
                                     trainable=True)
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name='relu')
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.variable_scope('conv5_1') as scope:
            kernel = tf.get_variable(name='weights',
                                     shape=[3, 3, 512, 512],
                                     initializer=tf.glorot_uniform_initializer(),
                                     regularizer=tf.nn.l2_loss,
                                     trainable=True)
            biases = tf.get_variable(name='biases',
                                     shape=[512],
                                     initializer=tf.zeros_initializer(),
                                     trainable=True)
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name='relu')
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.variable_scope('conv5_2') as scope:
            kernel = tf.get_variable(name='weights',
                                     shape=[3, 3, 512, 512],
                                     initializer=tf.glorot_uniform_initializer(),
                                     regularizer=tf.nn.l2_loss,
                                     trainable=True)
            biases = tf.get_variable(name='biases',
                                     shape=[512],
                                     initializer=tf.zeros_initializer(),
                                     trainable=True)
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name='relu')
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.variable_scope('conv5_3') as scope:
            kernel = tf.get_variable(name='weights',
                                     shape=[3, 3, 512, 512],
                                     initializer=tf.glorot_uniform_initializer(),
                                     regularizer=tf.nn.l2_loss,
                                     trainable=True)
            biases = tf.get_variable(name='biases',
                                     shape=[512],
                                     initializer=tf.zeros_initializer(),
                                     trainable=True)
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name='relu')
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool5')

        # fc1
        with tf.variable_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.get_variable(name='weights',
                                   shape=[shape, 4096],
                                   initializer=tf.glorot_uniform_initializer(),
                                   regularizer=tf.nn.l2_loss,
                                   trainable=True)
            fc1b = tf.get_variable(name='biases',
                                   shape=[4096],
                                   initializer=tf.zeros_initializer(),
                                   trainable=True)
            self.pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(self.pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.variable_scope('fc2') as scope:
            fc2w = tf.get_variable(name='weights',
                                   shape=[4096, 4096],
                                   initializer=tf.glorot_uniform_initializer(),
                                   regularizer=tf.nn.l2_loss,
                                   trainable=True)
            fc2b = tf.get_variable(name='biases',
                                   shape=[4096],
                                   initializer=tf.zeros_initializer(),
                                   trainable=True)
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.variable_scope('fc3') as scope:
            fc3w = tf.get_variable(name='weights',
                                   shape=[4096, 1000],
                                   initializer=tf.glorot_uniform_initializer(),
                                   regularizer=tf.nn.l2_loss,
                                   trainable=True)
            fc3b = tf.get_variable(name='biases',
                                   shape=[1000],
                                   initializer=tf.zeros_initializer(),
                                   trainable=True)
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]

        # Compute data loss and regularization loss
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.labels,
                logits=self.fc3l)
            data_loss = tf.reduce_mean(cross_entropy)
            reg_loss = tf.losses.get_regularization_loss()
            self.loss = tf.add(data_loss, reg_loss, name='data_and_reg_loss')
            correct_prediction = tf.equal(tf.argmax(self.fc3l, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('total_loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)

        with tf.name_scope('train_step'):
            global_step=tf.train.get_or_create_global_step()
            self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            self.train_step = self.optimizer.minimize(self.loss, global_step=global_step)

        with tf.name_scope('test'):
            self.softmax = tf.nn.softmax(self.fc3l)
