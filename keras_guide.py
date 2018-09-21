import tensorflow as tf
from tensorflow import keras

import numpy as np
import argparse

from data_utils import load_CIFAR10

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/tmp/cifar-10-batches-py', type=str,
                    help='Directory which contains the dataset')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    help='optimizer learning rate')
#parser.add_argument('--reg', default=0.0, type=float,
#                    help='Scalar giving L2 regularization strength.')

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
    X_train, y_train_1, X_test, y_test_1 = load_CIFAR10(input_dir)

    # Reshape the data into a 2D matrix. One row holds one sample.
    X_train = np.reshape(X_train,[X_train.shape[0], -1])
    X_test = np.reshape(X_test,[X_test.shape[0], -1])

    # One-hot encode the labels
    y_train = np.zeros((y_train_1.size,10))
    y_train[ np.arange(y_train_1.size), y_train_1] = 1
    y_test = np.zeros((y_test_1.size,10))
    y_test[ np.arange(y_test_1.size), y_test_1] = 1

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
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test
def get_random_data():
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 3))

    return data, labels

def main(argv):

    args = parser.parse_args(argv[1:])
    # 1.
    # Construct the model
    #
    inputs = keras.Input(shape=(32,))  # Returns a placeholder tensor
    
    # A layer instance is callable on a tensor, and returns a tensor.
    x = keras.layers.Dense(64, activation='relu')(inputs)
    x = keras.layers.Dense(64, activation='relu')(x)
    predictions = keras.layers.Dense(3, activation='softmax')(x)
    
    # Instantiate the model given inputs and outputs.
    cnn_model = keras.Model(inputs=inputs, outputs=predictions)
    
    # The compile step specifies the training configuration.
    cnn_model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])



    # 2. 
    # Load a dataset
    #
    #X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data(args.data_dir)
    #print('Train data shape: ', X_train.shape)
    #print('Train labels shape: ', y_train.shape)
    #print('Validation data shape: ', X_val.shape)
    #print('Validation labels shape: ', y_val.shape)
    #print('Test data shape: ', X_test.shape)
    #print('Test labels shape: ', y_test.shape)

    X_test, y_test = get_random_data()
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    ## Configure training set for TF Dataset
    #cifar10_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    #cifar10_train = cifar10_train.batch(args.batch_size).repeat()
    ## Configure validation set for TF Dataset
    #cifar10_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    #cifar10_val = cifar10_val.batch(args.batch_size).repeat()

    # 3.
    # Training
    #
    #cnn_model.fit(X_train, y_train, epochs=15,
    #              batch_size=args.batch_size,
    #              validation_data=(X_val,y_val))

    # 4.
    # Evaluation and Prediction
    #
    # Use an unseen test set
    eval_result = cnn_model.evaluate(X_test, y_test, args.batch_size)
    
    print(eval_result)
    print(cnn_model.metrics_names)

if __name__ == '__main__':
    tf.app.run()
