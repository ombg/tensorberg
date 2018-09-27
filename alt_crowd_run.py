# Working example for my blog post at:
# https://danijar.github.io/structuring-your-tensorflow-models
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# standard libs
import argparse
from tqdm import trange


# local libs
import helpers
import data_utils

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
        x = tf.layers.dense(inputs=x, units=10, activation=None)
        return x

    @helpers.define_scope
    def optimize(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.label,
            logits=self.prediction)

        total_loss = tf.reduce_mean(cross_entropy)
        helpers.variable_summaries(total_loss)

        # Global step is incremented whenever the graph sees a new batch
        global_step=tf.train.get_or_create_global_step()
        optimizer = tf.train.RMSPropOptimizer(0.03)

        return (total_loss,
            optimizer.minimize(total_loss, global_step=global_step))

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

    # Load data as numpy array
    data = data_utils.get_some_data(args.input_path,
                                    dataset_name=args.dataset_name)

    X_train, y_train, X_val, y_val, X_test, y_test = data

    # Initialize corresponding tf.placeholders and a tf.data.Dataset
    images_placeholder = tf.placeholder(X_train.dtype, [None, 3072])
    labels_placeholder = tf.placeholder(y_train.dtype, [None, 10])

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (images_placeholder, labels_placeholder)).batch(args.batch_size).repeat()
    # OMG, WHY, oh why you MUST call batch().repeat()??????????
    # TODO Here you could further preprocess your data !!

    # Create an uninitializaed iterator which can be reused with
    # different tf.data.Datasets as long as they have the same shape and type
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)


    #This gets the next element from the iterator.
    # TODO Is this one batch?
    features, labels = iterator.get_next(name='my_iterator')

    # Define model
    #model = Model(images_placeholder, labels_placeholder)
    model = Model(image=features, label=labels)

    # Until now the iterator is not bound to a dataset and is uninitialized.
    # Therefore, we now create init_ops. Later, a session runs these init_ops.
    merged = tf.summary.merge_all()
    global_step = tf.train.get_global_step()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(args.logdir + '_train', sess.graph)
    test_writer = tf.summary.FileWriter(args.logdir + '_test')
    #TODO why no sess.graph as last argument?

    tf.global_variables_initializer().run(session=sess)

    for i in t:
        # Load the dataset
        images, labels = mnist.test.images, mnist.test.labels

        summary_test, error, global_step_vl = sess.run(
            [merged, model.error, global_step],
            {images_placeholder: images, labels_placeholder: labels})

        test_writer.add_summary(summary_test, global_step=global_step_vl)

        print('Test error {:6.2f}%'.format(100 * error))
    # Use tqdm progress bar => trange()
    t = trange(1)
    for i in range(num_steps):
        images, labels = mnist.train.next_batch(args.batch_size)

        if i % 50 == 0:
            summary_train, loss_vl, global_step_vl = sess.run(
                [merged, model.optimize, global_step],
                {images_placeholder: images, labels_placeholder: labels})
            print('#{}: loss: {:6.2f}'.format(global_step_vl, loss_vl[0]))
            train_writer.add_summary(summary_train, global_step=global_step_vl)

        else:
            loss_vl = sess.run(
                [model.optimize],
                {images_placeholder: images, labels_placeholder: labels}) 

        t.set_postfix(loss='{:05.3f}'.format(loss_vl[0]))

    train_writer.close()
    test_writer.close()

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

    main()
