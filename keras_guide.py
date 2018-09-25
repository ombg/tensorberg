import tensorflow as tf
from tensorflow import keras

import numpy as np
import argparse

# My modules
import data_utils 
from ompy import ml, fileio

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

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def get_IMGDB_tfdataset(img_list_filename,
                        subtract_mean=True,
                        normalize_data=True):

    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string, channels=3)
        #image_resized = tf.image.resize_images(image_decoded, [28, 28])
        return image_decoded, label
    
    # A vector of filenames.
    images_list, labels_list = fileio.parse_imgdb_list(txt_list=img_list_filename)
    imgs = tf.constant(images_list)
    
    # `labels[i]` is the label for the image in `imgs[i]`.
    labels = tf.constant(labels_list)
    
    dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))
    dataset = dataset.map(_parse_function) 
    return dataset

def main(argv):

    run_id = np.random.randint(1e6,size=1)[0]
    print('run_id: {}'.format(run_id))
    args = parser.parse_args()
    print(args)
    # 1.
    # Construct the model
    #

    # Returns a placeholder tensor
    inputs = keras.Input(shape=(12288,),
                         batch_size=args.batch_size,
                         name='cifar_input_layer')
    
    x = keras.layers.Dense(
            units=400,
            kernel_regularizer=tf.keras.regularizers.l2(l=args.reg),
            activation='relu')(inputs)

    x = keras.layers.Dense(
            units=400,
            kernel_regularizer=tf.keras.regularizers.l2(l=args.reg),
            activation='relu')(x)

    x = keras.layers.Dense(
            units=400,
            kernel_regularizer=tf.keras.regularizers.l2(l=args.reg),
            activation='relu')(x)

    x = keras.layers.Dense(
            units=100,
            kernel_regularizer=tf.keras.regularizers.l2(l=args.reg),
            activation='relu')(x)

    predictions = keras.layers.Dense(
            units=4,
            kernel_regularizer=tf.keras.regularizers.l2(l=args.reg),
            activation='softmax')(x)
    
    # Instantiate the model given inputs and outputs.
    cnn_model = keras.Model(inputs=inputs, outputs=predictions)
    
    # The compile step specifies the training configuration.
    cnn_model.compile(optimizer= tf.keras.optimizers.SGD(lr=args.lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    # 2. 
    # Load a dataset
    #
    data = data_utils.get_some_data(
        input_path=args.input_path,
        input_path_imgdb_test=args.input_path_test,
        dataset_name=args.dataset_name,
        subtract_mean=True,
        normalize_data=True)

    data_utils.print_shape(data)
    X_train, y_train, X_val, y_val, X_test, y_test = data

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


    # 3.
    # Training
    #
    
    callbacks = [
        # Write TensorBoard logs to `./logs` directory
        keras.callbacks.TensorBoard(
            log_dir='./logs_model_0/run_' + str(run_id))
    ]
    #mask = np.random.randint(0,49000,size=100)
    #X_train = X_train[mask]
    #y_train = y_train[mask]
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
