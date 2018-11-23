from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np #TODO only because of load_weights_from_numpy

import operator
from functools import reduce

from models import layers

class AbstractRegressor(ABC):
    def __init__(self, config, data_loader=None):
        self.config = config

        if data_loader != None:
            self.data, self.gt_map = data_loader.get_input()
        else:
            self.data = tf.placeholder(tf.float32, [None, 224, 224, 3])
            self.gt_map = tf.placeholder(tf.float32, [None, 112, 112, 1])

        # TODO Necessary?
        tf.add_to_collection('inputs', self.data)
        tf.add_to_collection('inputs', self.gt_map)

        self.parameters = []  # save/load weights in here

        self._prediction = None
        self._loss = None
        self._optimize = None
        self._mae = None
        self._cm = None
        self._softmax = None
        
    def build_graph(self):
       self.prediction 
       self.loss
       self.optimize
       self.mae

    @abstractmethod
    def prediction(self):
        pass

    @property
    def loss(self):
        if self._loss == None:
            """
            Compute data loss and regularization loss
            """
            with tf.name_scope('loss'):
                data_loss = tf.losses.mean_squared_error(
                                    labels=self.gt_map,
                                    predictions=self.prediction)
                reg_loss = tf.losses.get_regularization_loss()
                self._loss = tf.add(data_loss, reg_loss, name='data_and_reg_loss')
                tf.summary.scalar('total_loss', self._loss)
        return self._loss

    @property
    def optimize(self):
        if self._optimize == None:
            with tf.name_scope('train_step'):
                global_step=tf.train.get_or_create_global_step()
                optimizer = tf.train.RMSPropOptimizer(self.config.learning_rate)
                self._optimize = optimizer.minimize(self.loss, global_step=global_step)
        return self._optimize

    @property
    def mae(self):
        if self._mae == None:
            with tf.name_scope('mae'):
                self._mae = tf.metrics.mean_absolute_error(
                                                  self.gt_map,
                                                  self.prediction,
                                                  name='mae_metric')
                tf.summary.scalar('mae', self._mae[0])
        return self._mae

    def load_weights_from_numpy(self, weights_file, sess, weights_to_load=None):
        weights = np.load(weights_file)

        if weights_to_load == None:
            weights_to_load = sorted(weights.keys())

        assert isinstance(weights_to_load, list)
        for i, k in enumerate(weights_to_load):
            assert isinstance(k, str)
            sess.run(self.parameters[i].assign(weights[k]))

class VggMod(AbstractRegressor):
    """
    Toy net, for testing. Do not use.
    """
    @property
    def prediction(self):
        if self._prediction == None:
            # zero-mean input
            with tf.name_scope('preprocess') as scope:
                mean = tf.constant([123.68, 116.779, 103.939],
                                   dtype=tf.float32,
                                   shape=[1, 1, 1, 3], name='zero_mean')
                centered_data = self.data - mean

            layer_name = 'conv1_1' # [?,224,224,64]
            conv1_1, kernel, biases = layers.conv(centered_data, 3, 3, 64, 1, 1,
                                                  name=layer_name, trainable=True)
            self.parameters += [kernel, biases]

            layer_name = 'conv1_2'
            conv1_2, kernel, biases = layers.conv(conv1_1, 3, 3, 64, 1, 1, 
                                                  name=layer_name, trainable=True)
            self.parameters += [kernel, biases]

            # pool1
            with tf.variable_scope('conv1_2') as scope:
                pool1 = tf.nn.max_pool(conv1_2,
                                       ksize=[1, 2, 2, 1], #TODO
                                       strides=[1, 2, 2, 1],
                                       padding='SAME',
                                       name=scope.name + 'pool1')


            layer_name = 'conv2_1'
            conv2_1, kernel, biases = layers.conv(pool1, 3, 3, 128, 1, 1,
                                                  name=layer_name, trainable=True)
            self.parameters += [kernel, biases]

            layer_name = 'conv2_2'
            conv2_2, kernel, biases = layers.conv(conv2_1, 3, 3, 128, 1, 1,
                                                  name=layer_name, trainable=True)
            self.parameters += [kernel, biases]

            # pool2
            with tf.variable_scope('conv2_2') as scope:
                pool2 = tf.nn.max_pool(conv2_2,
                                       ksize=[1, 2, 2, 1],
                                       strides=[1, 2, 2, 1],
                                       padding='SAME',
                                       name=scope.name + 'pool2')

            layer_name = 'conv3_1'
            conv3_1, kernel, biases = layers.conv(pool2, 3, 3, 256, 1, 1,
                                                  name=layer_name, trainable=True)
            self.parameters += [kernel, biases]

            layer_name = 'conv3_2'
            conv3_2, kernel, biases = layers.conv(conv3_1, 3, 3, 256, 1, 1,
                                                  name=layer_name, trainable=True)
            self.parameters += [kernel, biases]

            layer_name = 'conv3_3'
            conv3_3, kernel, biases = layers.conv(conv3_2, 3, 3, 256, 1, 1,
                                                  name=layer_name, trainable=True)
            self.parameters += [kernel, biases]
            # pool3
            with tf.variable_scope('conv3_3') as scope:
                pool3 = tf.nn.max_pool(conv3_3,
                                       ksize=[1, 2, 2, 1],
                                       strides=[1, 2, 2, 1],
                                       padding='SAME',
                                       name=scope.name + 'pool3')

            layer_name = 'conv4_1'
            conv4_1, kernel, biases = layers.conv(pool3, 3, 3, 512, 1, 1,
                                                  name=layer_name, trainable=True)
            self.parameters += [kernel, biases]

            layer_name = 'conv4_2'
            conv4_2, kernel, biases = layers.conv(conv4_1, 3, 3, 512, 1, 1,
                                                  name=layer_name, trainable=True)
            self.parameters += [kernel, biases]

            layer_name = 'conv4_3'
            conv4_3, kernel, biases = layers.conv(conv4_2, 3, 3, 512, 1, 1,
                                                  name=layer_name, trainable=True)
            self.parameters += [kernel, biases]
            # pool4
            with tf.variable_scope('conv4_3') as scope:
                pool4 = tf.nn.max_pool(conv4_3,
                                       ksize=[1, 2, 2, 1],
                                       strides=[1, 1, 1, 1],
                                       padding='SAME',
                                       name=scope.name + 'pool4')

            layer_name = 'conv5_1'
            conv5_1, kernel, biases = layers.conv(pool4, 3, 3, 512, 1, 1,
                                                  name=layer_name, trainable=True)
            self.parameters += [kernel, biases]

            layer_name = 'conv5_2'
            conv5_2, kernel, biases = layers.conv(conv5_1, 3, 3, 512, 1, 1,
                                                  name=layer_name, trainable=True)
            self.parameters += [kernel, biases]

            layer_name = 'conv5_3'
            conv5_3, kernel, biases = layers.conv(conv5_2, 3, 3, 512, 1, 1,
                                                  name=layer_name, trainable=True)
            self.parameters += [kernel, biases]

            concat_layer = tf.concat([pool3, pool4, conv5_3], axis=3)
            
            layer_name = 'conv6'
            conv6, kernel, biases = layers.conv(concat_layer, 3, 3, 64, 1, 1,
                                                  name=layer_name, trainable=True)
            self.parameters += [kernel, biases]
            layer_name = 'conv7'
            conv7, kernel, biases = layers.conv(conv6, 1, 1, 1, 1, 1,
                                                  name=layer_name, trainable=True)
            conv7_relu = tf.nn.leaky_relu(conv7, alpha=0.01, name='conv7_relu')
            self._prediction = conv7_relu
        return self._prediction
