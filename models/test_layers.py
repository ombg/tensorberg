import tensorflow as tf
import layers

class LayersTest(tf.test.TestCase):

    def test_fc(self):
        with self.test_session() as sess:
            # TODO Come up with better/useful testcase
            x = tf.zeros((50,10), dtype=tf.float32)
            expected_fc_out = tf.zeros((50,4),dtype=tf.float32, name='fc')
            actual_fc_out, weights, biases = layers.fc(x, 10, 4, name='fc', relu=False)
            sess.run(tf.initializers.variables([weights, biases]))
            self.assertAllEqual(actual_fc_out, expected_fc_out)

    def test_conv(self):
        with self.test_session():
            with self.assertRaises(NotImplementedError):
                layers.conv()

if __name__ == '__main__':
    tf.test.main()
