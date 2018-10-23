import tensorflow as tf

class FullyConnectedNet:
    def __init__(self, config, data_loader=None):


        self.config = config

        self.bottlenecks = None
        self.labels = None
        self.out_argmax = None
        self.loss = None
        self.accuracy = None
        self.optimizer = None
        self.train_step = None
        self.softmax = None
        self.parameters = []

        if data_loader != None:
            self.bottlenecks, self.labels = data_loader.get_input()
        else:
            self.images = tf.placeholder(tf.float32, [None, 224, 224, 3])
            self.labels = tf.placeholder(tf.int32, [None])

        self.build_model()

    def load_weights_from_numpy(self, weight_file, sess):
        tf.logging.info('Loading pre-trained weights...')
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            sess.run(self.parameters[i].assign(weights[k]))

    def build_model(self):

        """
        Inputs to the network
        """
        tf.add_to_collection('inputs', self.bottlenecks)
        tf.add_to_collection('inputs', self.labels)

        """
        Network Architecture
        """

        # fc3
        with tf.variable_scope('fc3') as scope:
            fc3w = tf.get_variable(name='weights',
                                   shape=[4096, 5],
                                   initializer=tf.glorot_uniform_initializer(),
                                   regularizer=tf.nn.l2_loss,
                                   trainable=True)
            fc3b = tf.get_variable(name='biases',
                                   shape=[5],
                                   initializer=tf.zeros_initializer(),
                                   trainable=True)
            self.fc3l = tf.nn.bias_add(tf.matmul(self.bottlenecks, fc3w), fc3b)
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
