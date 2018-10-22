import sys

sys.path.extend(['..'])
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from utils import datahandler

from configs.imagenet_classes import class_names

class Trainer:
    def __init__(self, sess, model, config, data_loader=None):
        """
        Constructing the trainer
        :param sess: tF.Session() instance
        :param model: The model instance
        :param config: config namespace which will contain all the configurations you have specified in the json
        :param data: The data loader if specified. 
        """
        # Assign all class attributes
        self.model = model
        self.config = config
        self.sess = sess
        self.data_loader = data_loader

        self.write_op = tf.summary.merge_all()
        # Initialize all variables of the graph
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def train(self):
        if self.data_loader == None:
            raise RuntimeError

        run_id = np.random.randint(1e6,size=1)[0]
        print('run_id: {}'.format(run_id))
        print('Initializing data...')
        
        global_step = tf.train.get_or_create_global_step()
    
        saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        # File writers for TensorBoard
        train_writer = tf.summary.FileWriter( 
            self.config.summary_dir + 'run_' + str(run_id) + '/train', self.sess.graph)
        val_writer = tf.summary.FileWriter( 
            self.config.summary_dir + 'run_' + str(run_id) + '/val', self.sess.graph)
    
        tf.logging.info(' Starting training now!')
        for i in range(self.config.num_epochs):
    
            # Initialize iterator with training data
            self.data_loader.initialize_train(self.sess)
            
            #Do not monitor, just train
            for _ in range(10):
                self.sess.run(self.model.train_step)
    
            # Monitor the training every epoch
            fetches = [self.model.train_step,
                       self.model.loss,
                       self.model.accuracy,
                       self.write_op,
                       global_step]
            _,loss_vl, train_acc, summary_train, global_step_vl = self.sess.run(fetches)
            train_writer.add_summary(summary_train, global_step=global_step_vl)
            train_writer.flush()
            #Check how it goes with validation set
            self.data_loader.initialize_val(self.sess)
            fetches_val = [self.model.loss,
                           self.model.accuracy,
                           self.write_op,
                           global_step]
            val_loss, val_acc, summary_val, global_step_vl = self.sess.run(fetches_val)
            print(('#{}: train_loss: {:5.2f} train_acc: {:5.2f}%' 
                   ' val_loss: {:5.2f} val_acc: {:5.2f}%').format(
                global_step_vl,
                loss_vl, train_acc*100.0,
                val_loss, val_acc*100.0))
            val_writer.add_summary(summary_val, global_step=global_step_vl)
            val_writer.flush()

        save_path = saver.save(self.sess, self.config.checkpoint_dir + '/run_' + str(run_id))
        print('Model checkpoint saved to %s' % save_path)
    
        train_writer.close()
        val_writer.close()

    def test(self):
        if self.data_loader == None:
            raise RuntimeError
        tf.logging.info('Starting testing now!')
        # Load test dataset
        self.data_loader.initialize_test(self.sess)
        global_step = tf.train.get_or_create_global_step()
        try:
            while True:
                fetches = [self.model.softmax, self.model.accuracy, global_step]
                # Gets matrix [batch_size x num_classes] predictions
                class_probs, accuracy, global_step_vl = self.sess.run(fetches)
                print('#{}: test_acc: {:5.2f}%'.format(global_step_vl, acc * 100.0))
                # Get top five
                predictions_per_sample = (np.argsort(class_probs,axis=1)[::-1])[0:5]
                print(predictions_per_sample)
                #print(class_names[p], prob[p])
        except tf.errors.OutOfRangeError:
            pass

    def predict(self, image_path, num_images=1):

        if num_images == 1:
            img1 = imread(image_path, mode='RGB')
            img1 = imresize(img1, (224, 224))
        else:
            raise NotImplementedError

        prob = self.sess.run(self.model.softmax,
                                feed_dict={self.model.images: [img1]})[0]
        preds = (np.argsort(prob)[::-1])[0:5]
        for p in preds:
            print(class_names[p], prob[p])

    def create_bottlenecks(self, subset):
        if self.data_loader == None:
            raise RuntimeError
        tf.logging.info('Creating bottlenecks at ' + self.config.bottleneck_dir)
        global_step = tf.train.get_or_create_global_step()
        # Load dataset
        if subset == 'training':
            self.data_loader.initialize_train(self.sess)
            tf.logging.info('Reading training subset!')
        elif subset == 'validation':
            self.data_loader.initialize_val(self.sess)
            tf.logging.info('Reading validation subset!')
        else:
            raise NotImplementedError

        try:
            bottleneck_paths = datahandler.get_bottlenecks_from_ord_dict(
                                              self.data_loader.image_lists,
                                              self.config.bottleneck_dir,   
                                              subset=subset)

            for i, bn_path in enumerate(bottleneck_paths):
                # fc2 penultimate layer, ReLu'd
                fetches = [self.model.fc2]
                #Get bottleneck feature vector for an image
                bottleneck_values = self.sess.run(fetches)[0]
                bottleneck_string = ','.join(str(x) for x in bottleneck_values.squeeze())
                with open(bn_path, 'w') as bottleneck_file:
                    bottleneck_file.write(bottleneck_string)
                tf.logging.info('#{}: Written bottleneck to {}.'.format(i, bn_path))

        except KeyError:
            tf.logging.error('Bottlenecks not created.')
        except OSError as e:
            print(e)
