import tensorflow as tf

class FullyConnectedNet:
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

        with tf.name_scope('input'):
            bottleneck_input = tf.placeholder_with_default(
                                    bottleneck_tensor,
                                    shape=[None, bottleneck_tensor_size],
                                    name='BottleneckInputPlaceholder')

            ground_truth_input = tf.placeholder(
                tf.int64, [batch_size], name='GroundTruthInput')

        self.build_model()

    def build_model(self):

        """
        Inputs to the network
        """
        tf.add_to_collection('inputs', self.images)
        tf.add_to_collection('inputs', self.labels)

        """
        Network Architecture
        """

  # Organizing the following ops so they are easier to see in TensorBoard.
  layer_name = 'final_retrain_ops'
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      initial_value = tf.truncated_normal(
          [bottleneck_tensor_size, class_count], stddev=0.001)
      layer_weights = tf.Variable(initial_value, name='final_weights')
      variable_summaries(layer_weights)

    with tf.name_scope('biases'):
      layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
      variable_summaries(layer_biases)

    with tf.name_scope('Wx_plus_b'):
      logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
      tf.summary.histogram('pre_activations', logits)

  final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
