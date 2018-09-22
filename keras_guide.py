import tensorflow as tf
from tensorflow import keras

import numpy as np
import argparse

# My modules
import data_utils 
from ompy import ml

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/tmp/cifar-10-batches-py', type=str,
                    help='Directory which contains the dataset')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--lr', default=1e-2, type=float,
                    help='optimizer learning rate')
parser.add_argument('--reg', default=1e-2, type=float,
                    help='Scalar giving L2 regularization strength.')

def main(argv):

    run_id = np.random.randint(1e6,size=1)[0]
    print('run_id: {}'.format(run_id))
    args = parser.parse_args()
    print(args)
    # 1.
    # Construct the model
    #

    # Returns a placeholder tensor
    inputs = keras.Input(shape=(3072,),
                         batch_size=args.batch_size,
                         name='cifar_input_layer')
    
    x = keras.layers.Dense(
            units=100,
            kernel_initializer=keras.initializers.RandomNormal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=args.reg),
            activation=tf.keras.activations.relu)(inputs)

    x = keras.layers.Dense(
            units=100,
            kernel_initializer=keras.initializers.RandomNormal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=args.reg),
            activation=tf.keras.activations.relu)(x)

    x = keras.layers.Dense(
            units=100,
            kernel_initializer=keras.initializers.RandomNormal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=args.reg),
            activation=tf.keras.activations.relu)(x)

    x = keras.layers.Dense(
            units=100,
            kernel_initializer=keras.initializers.RandomNormal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=args.reg),
            activation=tf.keras.activations.relu)(x)

    predictions = keras.layers.Dense(
        units=10,
        kernel_initializer=keras.initializers.RandomNormal(),
        kernel_regularizer=tf.keras.regularizers.l2(l=args.reg),
        activation=tf.keras.activations.softmax)(x)
    
    # Instantiate the model given inputs and outputs.
    cnn_model = keras.Model(inputs=inputs, outputs=predictions)
    
    # The compile step specifies the training configuration.
    cnn_model.compile(optimizer= tf.keras.optimizers.SGD(lr=args.lr),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

    # 2. 
    # Load a dataset
    #
    cifar_data = data_utils.get_CIFAR10_data(args.data_dir,
                                             subtract_mean=True,
                                             normalize_data=True)

    data_utils.print_shape(cifar_data)
    X_train, y_train, X_val, y_val, X_test, y_test = cifar_data
    
    # For a fully-connected net, reshape the samples to single rows.
    X_train = np.reshape(X_train,[X_train.shape[0], -1])
    X_val = np.reshape(X_val,[X_val.shape[0], -1])
    X_test = np.reshape(X_test,[X_test.shape[0], -1])

    # One-hot encode the labels
    y_train = ml.makeonehot(y_train)
    y_val = ml.makeonehot(y_val)
    y_test = ml.makeonehot(y_test)
    
    # Alternatively, get some random data for sanity checks.
    #X_test, y_test = get_random_data()
    #data_utils.print_shape((X_test, y_test))

    ## Configure training set for TF Dataset (optional)
    #cifar10_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    #cifar10_train = cifar10_train.batch(args.batch_size).repeat()
    ## Configure validation set for TF Dataset
    #cifar10_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    #cifar10_val = cifar10_val.batch(args.batch_size).repeat()

    # 3.
    # Training
    #
    
    callbacks = [
        # Write TensorBoard logs to `./logs` directory
        keras.callbacks.TensorBoard(
            log_dir='./logs_cifar_model_0/run_' + str(run_id))
    ]
    cnn_model.fit(X_train, y_train, epochs=200,
                  batch_size=args.batch_size,
                  validation_data=(X_val,y_val),
                  callbacks=callbacks)

    # 4.
    # Evaluation and Prediction
    #
    # Use an unseen test set
    eval_result = cnn_model.evaluate(X_test, y_test, args.batch_size)
    
    print(eval_result)
    print(cnn_model.metrics_names)

if __name__ == '__main__':
    tf.app.run()
