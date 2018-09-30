# Working example for my blog post at:
# https://danijar.github.io/structuring-your-tensorflow-models
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# standard libs
import argparse

# local libs
import helpers
import data_utils

class Model:

    def __init__(self, image, label):
        self.image = image
        self.label = label
        self.prediction
        self.optimize
        self.evaluate

    @helpers.define_scope(initializer=tf.glorot_uniform_initializer())
    def prediction(self):

        x = self.image
        x = tf.layers.dense(inputs=x, units=200, activation=tf.nn.relu)
        x = tf.layers.dense(inputs=x, units=200, activation=tf.nn.relu)
        x = tf.layers.dense(inputs=x, units=200, activation=tf.nn.relu)
        x = tf.layers.dense(inputs=x, units=10, activation=None)
        return x


    @helpers.define_scope
    def optimize(self):

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.label,
            logits=self.prediction)

        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss', loss)

        # Global step is incremented whenever the graph sees a new batch
        global_step=tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(5e-4)

        return (loss,
                optimizer.minimize(loss, global_step=global_step))


    @helpers.define_scope
    def evaluate(self):
        truth_value = tf.math.equal(
            tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        accuracy = tf.reduce_mean(tf.cast(truth_value, tf.float32))
        # Log a lot of stuff for op `accuracy` in tensorboard.
        tf.summary.scalar('accuracy', accuracy)
        return accuracy

def main(argv):

    run_id = np.random.randint(1e6,size=1)[0]
    print('run_id: {}'.format(run_id))
    args = parser.parse_args(argv[1:])
    print(args)
    print('Loading data...')
    # Load data as numpy array
    data = data_utils.get_some_data(args.input_path,
                                    dataset_name=args.dataset_name)

    data_utils.print_shape(data)
    X_train, y_train, X_val, y_val, X_test, y_test = data

    print('Initializing data...')
    num_batches = len(X_train) // args.batch_size
    num_classes = len(np.unique(y_train))
    print('Number of batches per epoch: %d' % num_batches)

    train_dataset_x = tf.data.Dataset.from_tensor_slices(X_train)
    train_dataset_y = tf.data.Dataset.from_tensor_slices(y_train).map(
        lambda z: tf.one_hot(z, 10))
    train_dataset = tf.data.Dataset.zip((train_dataset_x, train_dataset_y))
    # TODO Use shuffle()?
    train_dataset = train_dataset.repeat().batch(args.batch_size)

    val_dataset_x = tf.data.Dataset.from_tensor_slices(X_val)
    val_dataset_y = tf.data.Dataset.from_tensor_slices(y_val).map(
        lambda z: tf.one_hot(z, 10))
    val_dataset = tf.data.Dataset.zip((val_dataset_x, val_dataset_y))
    # TODO Use shuffle()?
    val_dataset = val_dataset.repeat().batch(args.batch_size)

    # TODO Here you could further preprocess your data !!

    # Create an uninitializaed iterator which can be reused with
    # different tf.data.Datasets as long as they have the same shape and type
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)


    #This gets the next element from the iterator.
    # Note, it does not depend on train or test set.
    features, labels = iterator.get_next()

    # Until now the iterator is not bound to a dataset and is uninitialized.
    # Therefore, we now create init_ops. Later, a session runs these init_ops.
    train_init_op = iterator.make_initializer(train_dataset)
    val_init_op = iterator.make_initializer(val_dataset)

    #
    # Define a model
    print('Setting up the model...')
    model = Model(image=features, label=labels)

    global_step = tf.train.get_global_step()
    write_op = tf.summary.merge_all()


    #
    # Create the session
    sess = tf.Session()
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    train_writer = tf.summary.FileWriter( 
        args.logdir + '/run_' + str(run_id) + '/train', sess.graph)
    val_writer = tf.summary.FileWriter( 
        args.logdir + '/run_' + str(run_id) + '/val', sess.graph)
    print(' Starting training now!')
    for i in range(args.epochs):

        # Initialize iterator with training data
        sess.run(train_init_op)
        for _ in range(num_batches):
            #TODO global_step not needed here(?)
            sess.run([model.optimize, global_step])

        # Monitor the training after every epoch
        fetches = [model.optimize, model.evaluate, write_op, global_step]
        loss_vl, train_acc, summary_train, global_step_vl = sess.run(fetches)
        train_writer.add_summary(summary_train, global_step=global_step_vl)
        train_writer.flush()

        sess.run(val_init_op)
        fetches_val = [model.evaluate, write_op, global_step]
        val_acc, summary_val, global_step_vl = sess.run(fetches_val)
        print('#{}: loss: {:5.2f} train_acc: {:5.2f}% val_acc: {:5.2f}%'.format(
            global_step_vl,
            loss_vl[0],
            train_acc*100.0,
            val_acc*100.0))
        val_writer.add_summary(summary_val, global_step=global_step_vl)
        val_writer.flush()

    train_writer.close()
    val_writer.close()

if __name__ == '__main__':

    # Input arguments for setting up the model.
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_path', 
                        default='/tmp/cifar-10-batches-py/',
                        type=str,
                        help='Path which contains the dataset')
    
    parser.add_argument('--input_path_test', 
                        default=None,
                        type=str,
                        help=('Path which contains the test dataset.'
                              'Mandatory for IMGDB dataset.'))
    
    parser.add_argument('--dataset_name',
                        default='cifar',
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

    parser.add_argument('--epochs',
                        default=10,
                        type=int,
                        help='Train for this number of epochs.')

    # Starting the program
    tf.app.run()
    print('Done!')
