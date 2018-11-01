import tensorflow as tf

from models import layers

class FullyConnectedNet:
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

        self.parameters = []  # save/load weights here

        self._prediction = None
        self._loss = None
        self._optimize = None
        self._accuracy = None
        self._cm = None
        
    def build_graph(self):
       self.prediction 
       self.loss
       self.cm
       self.optimize
       self.accuracy

    @property
    def prediction(self):
        if self._prediction == None:
            # fc3
            fc3l, fc3w, fc3b = layers.fc(self.data,
                                         num_in=4096,
                                         num_out=4,
                                         name='output_layer',
                                         relu=False)
            self.parameters += [fc3w, fc3b]
            self._prediction = fc3l
            tf.summary.histogram('fc3logits', fc3l)
        return self._prediction

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
                softmax = tf.nn.softmax(self.prediction)
                tf.summary.histogram('softmax', softmax)
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

    def load_weights_from_numpy(self, weight_file, sess):
        tf.logging.info('Loading pre-trained weights...')
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            sess.run(self.parameters[i].assign(weights[k]))
