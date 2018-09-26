# Working example for my blog post at:
# https://danijar.github.io/structuring-your-tensorflow-models
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# standard libs
import argparse

# local libs
import helpers

class Model:

    def __init__(self, image, label):
        self.image = image
        self.label = label
        self.prediction
        self.optimize
        self.error

    @helpers.define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def prediction(self):
        x = self.image
        x = tf.layers.dense(inputs=x, units=200, activation=tf.nn.relu)
        x = tf.contrib.slim.fully_connected(x, 200)
        x = tf.contrib.slim.fully_connected(x, 200)
        x = tf.contrib.slim.fully_connected(x, 10, tf.nn.softmax)
        return x

    @helpers.define_scope
    def optimize(self):
        logprob = tf.log(self.prediction + 1e-12)
        cross_entropy = -tf.reduce_sum(self.label * logprob)
        optimizer = tf.train.RMSPropOptimizer(0.03)
        return optimizer.minimize(cross_entropy)

    @helpers.define_scope
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        mistakes = tf.reduce_mean(tf.cast(mistakes, tf.float32))
        # Log a lot of stuff for `mistakes` in tensorboard.
        helpers.variable_summaries(mistakes)
        return mistakes

def main():

#    run_id = np.random.randint(1e6,size=1)[0]
#    print('run_id: {}'.format(run_id))
    args = parser.parse_args()
    print(args)

    mnist = input_data.read_data_sets('./data/mnist/', one_hot=True)
    image = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])

    model = Model(image, label)

    merged = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(args.logdir, sess.graph)
    test_writer = tf.summary.FileWriter(args.logdir) #TODO why no sess.graph?

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

    # Input arguments for setting up the model.
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_path', 
                        default='/tmp/cl_0123_ps_64_dm_55_sam_799_0_ppm.txt',
                        type=str,
                        help='Path which contains the dataset')
    
    parser.add_argument('--input_path_test', 
                        default='/tmp/cl_0123_ps_64_dm_55_sam_799_1_ppm.txt',
                        type=str,
                        help=('Path which contains the test dataset.'
                              'Mandatory for IMGDB dataset.'))
    
    parser.add_argument('--dataset_name',
                        default='imgdb',
                        type=str,
                        help='Name of the dataset. Supported: CIFAR-10 or IMGDB')
    
    parser.add_argument('--batch_size', 
                        default=100,
                        type=int,
                        help='batch size')
    
    parser.add_argument('--lr',
                        default=1e-2,
                        type=float,
                        help='optimizer learning rate')
    
    parser.add_argument('--reg',
                        default=1e-2,
                        type=float,
                        help='Scalar giving L2 regularization strength.')
    
    parser.add_argument('--logdir',
                        default='./logs/',
                        type=str,
                        help='Log directory for TensorBoard.')

    main()
