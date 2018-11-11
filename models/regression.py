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
            self.data, self.labels = data_loader.get_input()
        else:
            self.data = tf.placeholder(tf.float32, [None, 224, 224, 3])
            self.labels = tf.placeholder(tf.int32, [None])

        # TODO Necessary?
        tf.add_to_collection('inputs', self.data)
        tf.add_to_collection('inputs', self.labels)

        self.parameters = []  # save/load weights in here

        self._prediction = None
        self._loss = None
        self._optimize = None
        self._accuracy = None
        self._cm = None
        self._softmax = None
        
    def build_graph(self):
       self.prediction 
       self.loss
       self.softmax
       self.cm
       self.optimize
       self.accuracy

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
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                                    labels=self.labels,
                                    logits=self.prediction)
                data_loss = tf.reduce_mean(cross_entropy)
                reg_loss = tf.losses.get_regularization_loss()
                self._loss = tf.add(data_loss, reg_loss, name='data_and_reg_loss')
                tf.summary.scalar('total_loss', self._loss)
        return self._loss

    @property
    def optimize(self):
        if self._optimize == None:
            with tf.name_scope('train_step'):
                global_step=tf.train.get_or_create_global_step()
                optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
                self._optimize = optimizer.minimize(self.loss, global_step=global_step)
        return self._optimize

    @property
    def accuracy(self):
        if self._accuracy == None:
            with tf.name_scope('accuracy'):
                pred = tf.argmax(self.prediction, axis=1)
                lbl  = tf.argmax(self.labels, axis=1)
                is_correct = tf.equal(pred, lbl)
                self._accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
                tf.summary.scalar('accuracy', self._accuracy)
        return self._accuracy

    @property
    def cm(self):
        if self._cm == None:
            pred = tf.argmax(self.prediction, axis=1)
            lbl  = tf.argmax(self.labels, axis=1)
            self._cm = tf.confusion_matrix(lbl, pred)
        return self._cm

    @property
    def softmax(self):
        if self._softmax == None:
            self._softmax = tf.nn.softmax(self.prediction)
            tf.summary.histogram('softmax', self._softmax)
        return self._cm

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
                                                  name=layer_name, trainable=False)
            self.parameters += [kernel, biases]

            layer_name = 'conv1_2'
            conv1_2, kernel, biases = layers.conv(conv1_1, 3, 3, 64, 1, 1, 
                                                  name=layer_name, trainable=False)
            self.parameters += [kernel, biases]

            # pool1
            with tf.variable_scope('conv1_2') as scope:
                pool1 = tf.nn.max_pool(conv1_2,
                                       ksize=[1, 4, 4, 1], #TODO
                                       strides=[1, 4, 4, 1],
                                       padding='SAME',
                                       name=scope.name + 'pool1')

            # flatten the output volume
            shape = reduce(operator.mul, pool1.get_shape()[1:].as_list(), 1)
            pool1_flat = tf.reshape(pool1, [-1, shape])

            layer_name = 'fc2'
            fc2l, fc2w, fc2b = layers.fc(pool1_flat,
                                         num_in=shape,
                                         units=1024,
                                         name=layer_name,
                                         relu=True)
            tf.summary.histogram(layer_name, fc2l)

            layer_name = 'fc3'
            fc3l, fc3w, fc3b = layers.fc(fc2l,
                                         num_in=fc2l.get_shape()[1],
                                         units=5,
                                         name=layer_name,
                                         relu=False)
            tf.summary.histogram(layer_name, fc3l)
            self._prediction = fc3l
        return self._prediction