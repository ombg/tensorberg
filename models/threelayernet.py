import tensorflow as tf

# local libs
from models import helpers

class ThreeLayerNet:

    def __init__(self, image, label):
        self.image = image
        self.label = label
        self.prediction
        self.optimize
        self.evaluate

    @helpers.define_scope(initializer=tf.glorot_uniform_initializer())
    def prediction(self):

        x = self.image
        conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[7,7],
                                 strides=[1,1], padding='valid',
                                 activation=tf.nn.relu,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                 name='conv1')

        # Plot weights and images in Tensorboard.
        #kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'prediction/conv1/kernel')[0]
        #bias   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'prediction/conv1/bias')[0]

        # Careful, generates huge log files and causes tensorboard 
        # to run out of memory when dataset is not tiny.
#        with tf.name_scope('conv_layer_1'):
#            image_shaped_input = tf.reshape(x, [-1, 32, 32, 3])
#            tf.summary.image('input', image_shaped_input, 10)
#            kernel_shaped_input = tf.reshape(kernel, [-1, 7, 7, 3])
#            tf.summary.image('kernel', kernel_shaped_input, 32)
        
        pool_out = tf.layers.max_pooling2d(inputs=conv1,
                        pool_size=[2,2],
                        strides=[2,2], 
                        padding='VALID',
                        name='pool1')
        
        # output volume after pooling and flattened.
        dim_pool_out_flat = pool_out.get_shape()[1] * pool_out.get_shape()[2] * pool_out.get_shape()[3]
        
        # FC layer 1
        pool_out_flat = tf.reshape(pool_out,[-1, int(dim_pool_out_flat)])
        y = tf.layers.dense(inputs=pool_out_flat,
                            units=1024,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                            activation=tf.nn.relu)

        # FC layer 2 - output layer
        y = tf.layers.dense(inputs=y,
                            units=10,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                            activation=None)
        return y


    @helpers.define_scope
    def optimize(self):

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.label,
            logits=self.prediction)

        data_loss = tf.reduce_mean(cross_entropy)

        reg_loss = tf.losses.get_regularization_loss()
        loss = tf.add(data_loss, reg_loss, name='data_and_reg_loss')
        # Global step is incremented whenever the graph sees a new batch
        global_step=tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(5e-4)

        tf.summary.scalar('data_loss', data_loss)
        tf.summary.scalar('reg_loss', reg_loss)
        tf.summary.scalar('loss', loss)
        return (loss,
                optimizer.minimize(loss, global_step=global_step))


    @helpers.define_scope
    def evaluate(self):
        truth_value = tf.math.equal(
            tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        accuracy = tf.reduce_mean(tf.cast(truth_value, tf.float32))
        # Log a lot of stuff for op `accuracy` in tensorboard.
        #tf.summary.scalar('accuracy', accuracy)
        return accuracy

