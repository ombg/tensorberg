import tensorflow as tf
from tensorflow import keras

import numpy as np
import argparse

from data_utils import load_CIFAR10

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/tmp/cifar-10-batches-py', type=str,
                    help='Directory which contains the dataset')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    help='optimizer learning rate')
#parser.add_argument('--reg', default=0.0, type=float,
#                    help='Scalar giving L2 regularization strength.')
parser.add_argument('--train_steps', default=20000, type=int,
                    help='number of training steps')
parser.add_argument('--overfit', default=False, type=bool,
                    help=('If true, it overfits a small subset'
                          ' of the data as a sanity check.'))

def get_CIFAR10_data(input_dir,
                     num_training=49000, 
                     num_validation=1000, 
                     num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    X_train, y_train, X_test, y_test = load_CIFAR10(input_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    # mean_image = np.mean(X_train, axis=0)
    # X_train -= mean_image
    # X_val -= mean_image
    # X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test

def main(argv):

    args = parser.parse_args(argv[1:])
    # 1.
    # Construct the model
    #
    cnn_model = keras.Sequential()
    # fc1
    cnn_model.add(keras.layers.Dense(64, activation='relu'))
    # fc2
    cnn_model.add(keras.layers.Dense(64, activation='relu'))
    # softmax
    cnn_model.add(keras.layers.Dense(10, activation='softmax'))

    # Configure the model's training process
    cnn_model.compile(optimizer=tf.train.AdamOptimizer(args.learning_rate),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=[keras.metrics.categorical_accuracy])



    # 2. 
    # Load a dataset
    #
    X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data(args.data_dir)
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)


    cifar10_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    cnn_model.fit(cifar10_data, epochs=15, steps_per_epoch=30)

if __name__ == '__main__':
    tf.app.run()
