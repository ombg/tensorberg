import tensorflow as tf
import numpy as np
import argparse

# local libs
import data_utils
from models import threelayernet

def train(data, args):
    X_train, y_train, X_val, y_val, X_test, y_test = data
#    idx_overfit=np.random.choice(len(X_train),size=100,replace=False)
#    X_train= X_train[idx_overfit]
#    y_train= y_train[idx_overfit]
    run_id = np.random.randint(1e6,size=1)[0]
    print('run_id: {}'.format(run_id))
    print('Initializing data...')
    num_batches = len(X_train) // args.batch_size
    num_classes = len(np.unique(y_train))
    print('Number of batches per epoch: %d' % num_batches)

    train_dataset_x = tf.data.Dataset.from_tensor_slices(X_train)
    train_dataset_y = tf.data.Dataset.from_tensor_slices(y_train).map(
        lambda z: tf.one_hot(z, num_classes))
    train_dataset = tf.data.Dataset.zip((train_dataset_x, train_dataset_y))
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train))
    train_dataset = train_dataset.repeat().batch(args.batch_size)

    val_dataset_x = tf.data.Dataset.from_tensor_slices(X_val)
    val_dataset_y = tf.data.Dataset.from_tensor_slices(y_val).map(
        lambda z: tf.one_hot(z, num_classes))
    val_dataset = tf.data.Dataset.zip((val_dataset_x, val_dataset_y))
    val_dataset = val_dataset.shuffle(buffer_size=len(X_val))
    val_dataset = val_dataset.repeat().batch(args.batch_size)

    # TODO Here you could further preprocess your data !!
    # Therefore have a look at ioutils.py

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
    model = threelayernet.ThreeLayerNet(image=features, label=labels)

    global_step = tf.train.get_global_step()
    write_op = tf.summary.merge_all()


    #
    # Create the session
    sess = tf.Session()
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # File writers for TensorBoard
    train_writer = tf.summary.FileWriter( 
        args.logdir + '/run_' + str(run_id) + '/train', sess.graph)
    val_writer = tf.summary.FileWriter( 
        args.logdir + '/run_' + str(run_id) + '/val', sess.graph)

    # The Saver() let's us save the weights to disk
    saver = tf.train.Saver()

    print(' Starting training now!')
    for i in range(args.epochs):

        # Initialize iterator with training data
        sess.run(train_init_op)
        
        #Do not monitor, just train
        for _ in range(num_batches):
            sess.run(model.optimize)

        # Monitor the training 
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
    save_path = saver.save(sess, './no_sync/model_checkpoints/run_' + str(run_id))
    print('Model checkpoint saved to %s' % save_path)

def evaluate(data, args):
    _, _, _, _, X_test, y_test = data
    print('Initializing test data...')
    num_batches = len(X_test) // args.batch_size
    num_classes = len(np.unique(y_test))
    print('Number of batches per epoch: %d' % num_batches)

    test_dataset_x = tf.data.Dataset.from_tensor_slices(X_test)
    test_dataset_y = tf.data.Dataset.from_tensor_slices(y_test).map(
        lambda z: tf.one_hot(z, num_classes))
    test_dataset = tf.data.Dataset.zip((test_dataset_x, test_dataset_y))
    test_dataset = test_dataset.shuffle(buffer_size=len(X_test))
    test_dataset = test_dataset.batch(args.batch_size)

    features, labels = test_dataset.make_one_shot_iterator().get_next()

    #
    # Define a model
    print('Setting up the model...')
    model = threelayernet.ThreeLayerNet(image=features, label=labels)
    global_step = tf.train.get_global_step()
    
    # The Saver() let's us load the weights from disk. (what a name...)
    saver = tf.train.Saver()
    #
    # Create the session
    sess = tf.Session()

    #Restor variables from disk - no initialization necessary.
    saver.restore(sess, args.checkpointdir)
    print('Model weights loaded.')

    try:
        while True:
            acc, global_step_vl = sess.run([model.evaluate, global_step])
            print('#{}: test_acc: {:5.2f}%'.format(global_step_vl, acc * 100.0))
    except tf.errors.OutOfRangeError:
        pass

def main(argv):

    args = parser.parse_args(argv[1:])
    print(args)

    print('Loading data...')
    # Load data as numpy array
    data = data_utils.get_some_data(
        args.input_path,
        input_path_imgdb_test=args.input_path_test,
        dataset_name=args.dataset_name,
        channels_first=False,
        reshape_data=False)

    data_utils.print_shape(data)
    #train(data, args)
    evaluate(data,args)

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

    parser.add_argument('--checkpointdir',
                        default=None, 
                        type=str,
                        help='Checkpoint directory to load a trained model from.')

    parser.add_argument('--epochs',
                        default=10,
                        type=int,
                        help='Train for this number of epochs.')

    # Starting the program
    tf.app.run()
    print('Done!')
