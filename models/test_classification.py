import functools
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from classification import ToyModel
from classification import FullyConnectedNet
from utils import config

class TestModels(tf.test.TestCase):
    def setUp(self):
        # Data is fed as numpy arrays into the model
        self.mnist = input_data.read_data_sets('./mnist/', one_hot=True)

        import json
        testcase_config_filename = '/tmp/testcase_config.json'
        testcase_config = {
            "data_path":"/tmp/cifar_tfrecord/",
            "is_training": "True",
            "validation_percentage": "10",
            "testing_percentage": "10",
            "dataset_name": "cifar10",
            "work_dir": "/tmp/work_dir/",
            "exp_name": "cifar10_classifier",
            "checkpoint_to_restore": "run_46505",
            "num_epochs": 5,
            "learning_rate": 0.0005,
            "batch_size": 32,
            "max_to_keep":5
        }
        with open(testcase_config_filename, 'w') as f:
            json.dump(testcase_config, f)
        self.testcase_config_reloaded = config.process_config(testcase_config_filename)

    def test_toynet(self):
        image = tf.placeholder(tf.float32, [None, 784])
        label = tf.placeholder(tf.float32, [None, 10])
        model = ToyModel(image, label)
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            for _ in range(10):
                images, labels = self.mnist.test.images, self.mnist.test.labels
                error = sess.run(model.error, {image: self.mnist.test.images, label: self.mnist.test.labels})
                print('Test error {:6.2f}%'.format(100 * error))
                for i in range(60):
                    images, labels = self.mnist.train.next_batch(100)
                    if i == 0:
                        loss, _ = sess.run([model.loss, model.optimize], {image: images, label: labels})
                        print('Training set loss: {:6.2f}'.format(loss))
                    else:
                        sess.run(model.optimize, {image: images, label: labels})

    def test_fullyconnectednet(self):
        model = FullyConnectedNet(self.testcase_config_reloaded)
        model.build_graph()
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            for _ in range(10):
                accuracy = sess.run(model.accuracy, {model.data: self.mnist.test.images, model.labels: self.mnist.test.labels})
                print('Test accuracy {:6.2f}%'.format(100 * accuracy))
                for i in range(60):
                    images, labels = self.mnist.train.next_batch(100)
                    if i == 0:
                        loss, _ = sess.run([model.loss, model.optimize], {model.data: images, model.labels: labels})
                        print('Training set loss: {:6.2f}'.format(loss))
                    else:
                        sess.run(model.optimize, {model.data: images, model.labels: labels})

if __name__ == '__main__':
    tf.test.main()
