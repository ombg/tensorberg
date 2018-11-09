import tensorflow as tf
import layers

class LayersTest(tf.test.TestCase):

    def test_fc_shape(self):
        with self.test_session() as sess:
            # TODO Come up with better/useful testcase
            x = tf.zeros((50,10), dtype=tf.float32)
            expected_fc_out = tf.zeros((50,4),dtype=tf.float32, name='expectedout')
            actual_fc_out, weights, biases = layers.fc(x, 10, 4, name='fc', relu=False)
            sess.run(tf.initializers.variables([weights, biases]))
            self.assertAllEqual(tf.shape(actual_fc_out), tf.shape(expected_fc_out))

    def test_conv_shape(self):
        with self.test_session() as sess:
            x = tf.zeros((32,227,227,3), dtype=tf.float32)
            expected_conv_out = tf.zeros((32,55,55,96),dtype=tf.float32, name='expectedout')
            actual_conv_out, weights, biases = layers.conv(x, 11, 11, 96, 4, 4, name='convtest', padding='VALID', groups=1)
            sess.run(tf.initializers.variables([weights, biases]))
            self.assertAllEqual(tf.shape(actual_conv_out), tf.shape(expected_conv_out))

if __name__ == '__main__':
    tf.test.main()
