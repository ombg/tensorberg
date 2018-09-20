import tensorflow as tf
from tensorflow import keras

import numpy as np
import argparse

parser = argparse.ArgumentParser()
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

def main():
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
    cifar10_data = tf.data.Dataset.from_tensor_slices()

if __name__ == '__main__':
    tf.app.run()
