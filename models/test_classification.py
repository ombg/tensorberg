import functools
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from classification import ToyModel

def main():
    mnist = input_data.read_data_sets('./mnist/', one_hot=True)
    image = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])
    model = ToyModel(image, label)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for _ in range(10):
        images, labels = mnist.test.images, mnist.test.labels
        error = sess.run(model.error, {image: images, label: labels})
        print('Test error {:6.2f}%'.format(100 * error))
        for i in range(60):
            images, labels = mnist.train.next_batch(100)
            if i == 0:
                loss, _ = sess.run([model.loss, model.optimize], {image: images, label: labels})
                print('Training set loss: {:6.2f}'.format(loss))
            else:
                sess.run(model.optimize, {image: images, label: labels})


if __name__ == '__main__':
    main()
