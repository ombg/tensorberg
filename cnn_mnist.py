"""
Based on TensorFlow tutorial
 'Build a Convolutional Neural Network using Estimators'
The well documented example code can be found here:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/layers/cnn_mnist.py
"""
import numpy as np
import tensorflow as tf

# Necessary for tf.train.LoggingTensorHook()
tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):

    # Input layer
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])
    
    # conv1: Conv layer with zero-padding and ReLU activations
    # padding='same': Use padding to have size output == size input
    # Size of output volume: 28x28x32 
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32,
                             kernel_size=[5,5], padding='same',
                             activation=tf.nn.relu)
                             
    # pool1: 2x2 pooling w/o overlap
    # Size of output volume: 14x14x32
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2],
                                    strides=[2,2])

    # conv2: output volume: 14x14x64
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64,
                             kernel_size=[5,5], padding='same',
                             activation=tf.nn.relu)

    # pool2: output volume 7x7x64
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2],
                                    strides=[2,2])

    # Flatten pooling layer in order to feed it to the 1st FC layer / dense layer
    # output volume (2D): [batchsize, 7*7*64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # fc1: weight matrix shape: [7*7*64,1024]
    fc1 = tf.layers.dense(inputs=pool2_flat, units=1024,
                          activation=tf.nn.relu)

    # Apply dropout on fc1. Switch off 40% during training.
    # output volume (2D): [batchsize, 1024]
    fc1_dropout = tf.layers.dropout(
                        inputs=fc1,
                        rate=0.4,
                        training = mode == tf.estimator.ModeKeys.TRAIN)

    # fc2: logits layer, no ReLU! TODO: units ok?
    fc2 = tf.layers.dense(inputs=fc1_dropout, units=labels.shape[0])

    predictions = {
        'classes': tf.argmax(input=fc2, axis=1),
        'probabilities': tf.nn.softmax(fc2, name='softmax_tensor')
    }

    # PREDICT MODE - We run the whole function in PREDICT mode,
    # no loss calculations necessary
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Starting here, we run the whole function in TRAIN or EVAL mode,
    # so we calculate the loss.
    # This function returns the average over the whole batch.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=fc2)

    # TRAIN MODE - Configure the Training Op
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(
            loss=loss,
            # Needed for TensorBoard
            global_step=tf.train.get_global_step())
        # The estimator API requires cnn_model_fn() to return EstimatorSpecs
        # Only then the Estimator knows how to compute the loss and the gradient.
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # EVAL MODE - Add evaluation metrics.
    # You can add multiple metrics. MESA distance?
    eval_metric_ops = {
        'accuracy' : tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes'])
        }
    # Make it available on TensorBoard TODO
    #tf.summary.scalar('accuracy', accuracy[1])
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops)


def main(unused_argv):

    # 1.
    # Load training and evaluate data
    #

    mnist = tf.contrib.learn.dataset.load_data_set('mnist')

    # Returns a numpy array shape: (55000,28,28,1)? TODO
    train_data = mnist.train.images
    train_labels = np.asarray( mnist.train.labels, dtype=np.int32 )

    # Returns a numpy array shape: (10000,28,28,1)? TODO
    eval_data = mnist.test.images
    eval_labels = np.asarray( mnist.test.labels, dtype=np.int32 )

    # 2.
    # Instaniate a TensorFlow Estimator class.
    #

    # It is a high-level representation of 
    # model training, evaluation and inference
    # Could be a classifier or a regressor.
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir='/tmp/mnist_convnet_model')

    # 3. (optional)
    # Set up logging to keep track of things
    #

    # This dictionary contains all tensors you want to log.
    # Feel free to choose meaningful dictionary keys.
    # This key points to an existing tensor, the 'softmax_tensor'.
    tensors_to_log = {'softmax_values': 'softmax_tensor'}

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # 4.
    # Train the model
    #

    # Set up the train-input function using train_data
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': train_data},
            y=train_labels,
            batch_size=100,
            num_epochs=None, # None == run forever
            shuffle=True)

    # Start training, using train_input_fn and logging_hook
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000, # stop after 20000 steps. TODO What if num_epochs=1?
        hooks=[logging_hook])

    # 5.
    # Evaluate the model
    #

    # Set up the evaluate-input function using eval_data
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': eval_data},
            y=eval_labels, # no batchsize specified
            num_epochs=1, # Makes sense. Test exactly ones on every sample.
            shuffle=False) # Apparently

    # Start evaluation and print the results
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
  tf.app.run()
