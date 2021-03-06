from abc import ABC, abstractmethod
import tensorflow as tf

from models import layers

class ToyModel:
    """A simple toy model. For test purposes it is defined in the constructor.
    Use decorators for more complex models.
    """

    def __init__(self, data, label, num_features, num_classes):

        hidden1, _, _ = layers.fc(data, num_features, 100, name='hidden1',log_weights=False)
        logits, _, _ = layers.fc(hidden1, 100, num_classes, name='logits', relu=False, log_weights=False)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                            labels=label,
                            logits=logits)
        data_loss = tf.reduce_mean(cross_entropy)
        reg_loss = 1e-3 * tf.losses.get_regularization_loss()
        self._loss = tf.add(data_loss, reg_loss, name='data_and_reg_loss')
        global_step=tf.train.get_or_create_global_step()
        self._optimize = tf.train.RMSPropOptimizer(0.03).minimize(self._loss, global_step=global_step)
        self._prediction = tf.nn.softmax(logits)
        mistakes = tf.not_equal(
            tf.argmax(label, 1), tf.argmax(self._prediction, 1))
        self._error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @property
    def prediction(self):
        return self._prediction

    @property
    def optimize(self):
        return self._optimize

    @property
    def error(self):
        return self._error

    @property
    def loss(self):
        return self._loss

class AbstractNet(ABC):
    def __init__(self, config, data_loader=None):
        self.config = config

        if data_loader != None:
            self.data, self.labels = data_loader.get_input()
        else:
            self.data = tf.placeholder(tf.float32, [None, 32, 32, 3])
            self.labels = tf.placeholder(tf.int32, [None, 10])

        self.num_classes = self.labels.get_shape()[1].value
        tf.summary.image('orig_image',self.data)

        # TODO Necessary?
        tf.add_to_collection('inputs', self.data)
        tf.add_to_collection('inputs', self.labels)

        self.parameters = []  # save/load weights here

        self._prediction = None
        self._loss = None
        self._optimize = None
        self._accuracy = None
        self._cm = None
        self._softmax = None
        
    def build_graph(self):
        try:
            self.prediction 
            self.loss
            self.softmax
            self.cm
            self.optimize
            self.accuracy
        except MemoryError as err:
            tf.logging.warning(err.args)

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
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                                    labels=self.labels,
                                    logits=self.prediction)
                data_loss = tf.reduce_mean(cross_entropy)
                reg_loss = 1e-2 * tf.losses.get_regularization_loss()
                self._loss = tf.add(data_loss, reg_loss, name='data_and_reg_loss')
                tf.summary.scalar('data_loss', data_loss)
                tf.summary.scalar('reg_loss', reg_loss)
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
            self._cm = tf.confusion_matrix(lbl, pred, num_classes=self.num_classes)
        return self._cm

    @property
    def softmax(self):
        if self._softmax == None:
            self._softmax = tf.nn.softmax(self.prediction)
            tf.summary.histogram('softmax', self._softmax)
        return self._softmax

    def load_weights_from_numpy(self, weights_file, sess, weights_to_load=None):
        weights = np.load(weights_file)

        if weights_to_load == None:
            weights_to_load = sorted(weights.keys())

        assert isinstance(weights_to_load, list)
        for i, k in enumerate(weights_to_load):
            assert isinstance(k, str)
            sess.run(self.parameters[i].assign(weights[k]))

class FullyConnectedNet(AbstractNet):
    def __init__(self, config, data_loader=None):
        super().__init__(config, data_loader=data_loader)
        self.data = tf.layers.flatten(self.data)

    @property
    def prediction(self):
        if self._prediction == None:
            data_dim = self.data.get_shape()[1].value
            layer_name = 'fc1'
            fc1l, fc1w, fc1b = layers.fc(self.data,
                                         num_in=data_dim,
                                         units=80,
                                         name=layer_name,
                                         relu=True)
            self.parameters += [fc1w, fc1b]

            layer_name = 'fc2'
            fc2l, fc2w, fc2b = layers.fc(fc1l,
                                         num_in=fc1l.get_shape()[1].value,
                                         units=80,
                                         name=layer_name,
                                         relu=True)
            self.parameters += [fc2w, fc2b]

            layer_name = 'fc3'
            fc3l, fc3w, fc3b = layers.fc(fc2l,
                                         num_in=fc2l.get_shape()[1].value,
                                         units=80,
                                         name=layer_name,
                                         relu=True)
            self.parameters += [fc3w, fc3b]

            layer_name = 'fc4'
            fc4l, fc4w, fc4b = layers.fc(fc3l,
                                         num_in=fc3l.get_shape()[1].value,
                                         units=80,
                                         name=layer_name,
                                         relu=True)
            self.parameters += [fc4w, fc4b]

            layer_name = 'fc5'
            fc5l, fc5w, fc5b = layers.fc(fc4l,
                                         num_in=fc4l.get_shape()[1].value,
                                         units=80,
                                         name=layer_name,
                                         relu=True)
            self.parameters += [fc5w, fc5b]

            layer_name = 'fc6'
            fc6l, fc6w, fc6b = layers.fc(fc5l,
                                         num_in=fc5l.get_shape()[1].value,
                                         units=80,
                                         name=layer_name,
                                         relu=True)
            self.parameters += [fc6w, fc6b]

            layer_name = 'fc7'
            fc7l, fc7w, fc7b = layers.fc(fc6l,
                                         num_in=fc6l.get_shape()[1].value,
                                         units=80,
                                         name=layer_name,
                                         relu=True)
            self.parameters += [fc7w, fc7b]

            layer_name = 'fc8'
            fc8l, fc8w, fc8b = layers.fc(fc7l,
                                         num_in=fc7l.get_shape()[1].value,
                                         units=10,
                                         name=layer_name,
                                         relu=False)
            self.parameters += [fc8w, fc8b]

            self._prediction = fc8l

        return self._prediction

class OutputLayer(AbstractNet):
    @property
    def prediction(self):
        if self._prediction == None:
            data_dim = int(self.data.shape[1])
            layer_name = 'logits'
            fc1l, fc1w, fc1b = layers.fc(self.data,
                                         num_in=data_dim,
                                         units=5,
                                         name=layer_name,
                                         relu=False)
            tf.summary.histogram(layer_name, fc1l)
            self._prediction = fc1l

        return self._prediction
