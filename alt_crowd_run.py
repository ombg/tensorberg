# Working example for my blog post at:
# https://danijar.github.io/structuring-your-tensorflow-models
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import utils

class Model:

    def __init__(self, image, label):
        self.image = image
        self.label = label
        self.prediction
        self.optimize
        self.error

    @utils.define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def prediction(self):
        x = self.image
        x = tf.contrib.slim.fully_connected(x, 200)
        x = tf.contrib.slim.fully_connected(x, 200)
        x = tf.contrib.slim.fully_connected(x, 10, tf.nn.softmax)
        return x

    @utils.define_scope
    def optimize(self):
        logprob = tf.log(self.prediction + 1e-12)
        cross_entropy = -tf.reduce_sum(self.label * logprob)
        optimizer = tf.train.RMSPropOptimizer(0.03)
        return optimizer.minimize(cross_entropy)

    @utils.define_scope
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        mistakes = tf.reduce_mean(tf.cast(mistakes, tf.float32))
        tf.summary.scalar('mistakes', mistakes)
        return mistakes

def main():
    mnist = input_data.read_data_sets('./data/mnist/', one_hot=True)
    image = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])
    model = Model(image, label)
    merged = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter('./logs/mnist/train', sess.graph)
    test_writer = tf.summary.FileWriter('./logs/mnist/test') #TODO why no sess.graph?
    tf.global_variables_initializer().run(session=sess)

    for i in range(10):
        images, labels = mnist.test.images, mnist.test.labels
        summary, error = sess.run([merged, model.error], {image: images, label: labels})
        test_writer.add_summary(summary, i)
        print('Test error {:6.2f}%'.format(100 * error))
        for _ in range(60):
            images, labels = mnist.train.next_batch(100)
            sess.run(model.optimize, {image: images, label: labels})

    train_writer.close()
    test_writer.close()

if __name__ == '__main__':
  main()
