import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse

import data_utils

from crowd_two import complex_model
#from crowd_one import simple_model

parser = argparse.ArgumentParser()

parser.add_argument('--input-path', 
                    default='/tmp/cl_0123_ps_64_dm_55_sam_799_0_ppm.txt',
                    type=str,
                    help='Path which contains the dataset')

parser.add_argument('--input-path-test', 
                    default='/tmp/cl_0123_ps_64_dm_55_sam_799_1_ppm.txt',
                    type=str,
                    help=('Path which contains the test dataset.'
                          'Mandatory for IMGDB dataset.'))

parser.add_argument('--dataset-name',
                    default='imgdb',
                    type=str,
                    help='Name of the dataset. Supported: CIFAR-10 or IMGDB')

parser.add_argument('--batch-size', 
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

def run_model(session, X, y, is_training, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [loss_val,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training
    
    # counter 
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[idx].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables,feed_dict=feed_dict)
            
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                      .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
            iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
              .format(total_loss,total_correct,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.draw()
    return total_loss,total_correct

def main(argv):

    # clear old variables
    tf.reset_default_graph()

    # Some housekeeping
    run_id = np.random.randint(1e6,size=1)[0]
    print('run_id: {}'.format(run_id))
    args = parser.parse_args()
    print(args)
    
    # Load CIFAR-10 dataset
    X_train, y_train, X_val, y_val, X_test, y_test = data_utils.get_some_data(
        input_path=args.input_path,
        dataset_name=args.dataset_name,
        subtract_mean=True,
        normalize_data=True)
    
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)
    
    # setup input (e.g. the data that changes every batch)
    # The first dim is None, and gets sets automatically based on batch size fed in
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)
    
    # Specify the model you want to use
    y_out = complex_model(X,y,is_training)
    
    # define our loss
    #total_loss = tf.losses.hinge_loss(tf.one_hot(y,10),logits=y_out)
    total_loss = tf.losses.softmax_cross_entropy(tf.one_hot(y,10),logits=y_out)
    mean_loss = tf.reduce_mean(total_loss)
    
    # define our optimizer
    optimizer = tf.train.AdamOptimizer(args.lr) # select optimizer and set learning rate
    train_step = optimizer.minimize(mean_loss)
    
    # Start the session and invoke training and then validation.
    with tf.Session() as sess:
        with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0" 
            sess.run(tf.global_variables_initializer())
            print('Training')
            run_model(sess,
                      X=X,
                      y=y,
                      predict=y_out,
                      loss_val=mean_loss,
                      Xd=X_train,
                      yd=y_train,
                      epochs=3,
                      batch_size=args.batch_size,
                      print_every=100,
                      training=train_step,
                      plot_losses=True)
            print('Validation')
            run_model(sess,y_out,mean_loss,X_val,y_val,1,64)
            run_model(sess,
                      X=X,
                      y=y,
                      predict=y_out,
                      loss=mean_loss,
                      Xd=X_val,
                      yd=y_val,
                      epochs=1,
                      batch_size=args.batch_size,
                      print_every=100,
                      training=train_step,
                      plot_losses=True)
            print('Done!')
    
    plt.show()

if __name__ == '__main__':
    tf.app.run()
