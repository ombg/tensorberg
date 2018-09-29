# Working example for my blog post at:
# https://danijar.github.io/structuring-your-tensorflow-models
import tensorflow as tf
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

#    run_id = np.random.randint(1e6,size=1)[0]
#    print('run_id: {}'.format(run_id))
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
    # create a placeholder to dynamically switch between batch sizes
    batch_size_placeholder = tf.placeholder(tf.int64)
    # Initialize corresponding tf.placeholders and a tf.data.Dataset
    images_placeholder = tf.placeholder(X_train.dtype, [None, 3072])
    labels_placeholder = tf.placeholder(y_train.dtype, [None, 10])

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (images_placeholder, labels_placeholder))
    # Adding the batch_size as an outer dimension. Repeat after 1 epoch
    train_dataset = train_dataset.batch(args.batch_size).repeat()

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (images_placeholder, labels_placeholder)).batch(args.batch_size)

    # TODO Here you could further preprocess your data !!

    # Create an uninitializaed iterator which can be reused with
    # different tf.data.Datasets as long as they have the same shape and type
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)


    #This gets the next element from the iterator.
    # Note, it does not depend on train or test set.
    features, labels = iterator.get_next(name='my_iterator')

    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    # Define model
    #model = Model(images_placeholder, labels_placeholder)
    print('Setting up the model...')
    model = Model(image=features, label=labels)

    # Until now the iterator is not bound to a dataset and is uninitialized.
    # Therefore, we now create init_ops. Later, a session runs these init_ops.
    merged_sm = tf.summary.merge_all()
    global_step = tf.train.get_global_step()

    # Create the session
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(args.logdir + '_train', sess.graph)

    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    # Initialize iterator with training data
    # This is the correct way when using numpy arrays (which fit into memory)
    sess.run(train_init_op, feed_dict= { images_placeholder: X_train,
                                         labels_placeholder: y_train,
                                         batch_size_placeholder: args.batch_size})

    print(' Starting training now!')
    for i in range(args.epochs):

        for _ in range(num_batches):
            sess.run([model.optimize, global_step])

        # Monitor the training after every epoch
        fetches = [model.optimize, model.evaluate, merged_sm, global_step]
        loss_vl, train_acc, summary_train, global_step_vl = sess.run(fetches)
        train_writer.add_summary(summary_train, global_step=global_step_vl)

        print('#{}: loss: {:6.2f} train_acc: {:6.2f}%'.format(
            global_step_vl,
            loss_vl[0], train_acc*100.0))
    # Initialize iterator with test data
    sess.run(test_init_op, feed_dict= { images_placeholder: X_test,
                                        labels_placeholder: y_test,
                                        batch_size_placeholder: len(y_test) })
    # Test the model on test data set
    print('Test accuracy: {:6.2f}%'.format(sess.run(model.evaluate) * 100.0))
    train_writer.close()

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
