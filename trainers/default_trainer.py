import sys

sys.path.extend(['..'])
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from utils import datahandler
from tqdm import tqdm

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

        # Initialize all variables of the graph
        self.model.build_graph()
        self.write_op = tf.summary.merge_all()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def train(self):
        if self.data_loader == None:
            raise RuntimeError

        run_id = np.random.randint(1e6,size=1)[0]
        tf.logging.info('run_id: {}'.format(run_id))
        tf.logging.info('Initializing data...')
        
        global_step = tf.train.get_or_create_global_step()
    
        saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        # File writers for TensorBoard
        train_writer = tf.summary.FileWriter( 
            self.config.summary_dir + 'run_' + str(run_id) + '/train', self.sess.graph)
        val_writer = tf.summary.FileWriter( 
            self.config.summary_dir + 'run_' + str(run_id) + '/val', self.sess.graph)
    
        tf.logging.info('Training for {} epochs...'.format(self.config.num_epochs))
        for i in range(self.config.num_epochs):
            tf.logging.info('===== EPOCH {} ====='.format(i))
            # Initialize iterator with training data
            self.data_loader.initialize_train(self.sess)
            
            #Do not monitor, just train for one epoch
            for _ in tqdm(range(self.data_loader.num_batches), ascii=True, desc='epoch'):
                self.sess.run(self.model.optimize)
    
            # Monitor the training after every epoch
            fetches = [self.model.optimize,
                       self.model.loss,
                       self.model.accuracy,
                       self.write_op,
                       global_step]

            _,loss_vl, train_acc, summary_train, global_step_vl = self.sess.run(fetches)
            train_writer.add_summary(summary_train, global_step=global_step_vl)
            train_writer.flush()
            #Check how it goes with the validation set
            self.data_loader.initialize_val(self.sess)
            fetches_val = [self.model.loss,
                           self.model.accuracy,
                           self.write_op,
                           global_step]
            val_loss, val_acc, summary_val, global_step_vl = self.sess.run(fetches_val)
            tf.logging.info(('#{}: train_loss: {:5.2f} train_acc: {:5.2f}%' 
                                 ' val_loss: {:5.2f} val_acc: {:5.2f}%').format(
                                     global_step_vl,
                                     loss_vl, train_acc*100.0,
                                     val_loss, val_acc*100.0))

            val_writer.add_summary(summary_val, global_step=global_step_vl)
            val_writer.flush()

        save_path = saver.save(self.sess, self.config.checkpoint_dir + 'run_' + str(run_id))
        tf.logging.info('Model checkpoint saved to %s' % save_path)
    
        train_writer.close()
        val_writer.close()

    def test(self, checkpoint_dir=None):

        # Load parameters
        if checkpoint_dir != None:
            saver = tf.train.Saver()
            saver.restore(self.sess, checkpoint_dir)

        # Check if test data is loaded
        if self.data_loader == None:
            raise RuntimeError
        tf.logging.info('Starting testing now!')
        # Load test dataset
        self.data_loader.initialize_test(self.sess)
        accuracies = []
        num_classes = int(self.data_loader.test_dataset.output_shapes[1][1])
        confusion_matrix = np.zeros((num_classes, num_classes),dtype=int)
        try:
            while True:
                fetches = [self.model.cm, self.model.accuracy]
                # Gets matrix [batch_size x num_classes] predictions
                cm, acc = self.sess.run(fetches)
                tf.logging.info('Per batch average test_acc: {:5.2f}%'.format(acc * 100.0))
                accuracies.append(acc)
                confusion_matrix += cm
        except tf.errors.OutOfRangeError:
            pass
        accuracies = np.asarray(accuracies)
        tf.logging.info('Average accuracy of batch accuracies: {:5.2f}% (std: {})'.format(
                            np.mean(accuracies) * 100.0,
                            np.std(accuracies)))
        tf.logging.info('Confusion matrix:\n{}'.format(confusion_matrix))

    def predict(self, image_path, num_images=1):
        raise NotImplementedError
        #TODO
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
            bottleneck_paths = self.data_loader.get_bottleneck_filenames(
                                                    self.config.bottleneck_dir,   
                                                    subset=subset)

            for bn_path in tqdm(bottleneck_paths,ascii=True, desc='bottlenecks'):
                # Specify bottleneck layer here:
                fetches = [self.model.prediction]
                #Get bottleneck feature vector for an image
                bottleneck_values = self.sess.run(fetches)[0]
                bottleneck_string = ','.join(str(x) for x in bottleneck_values.squeeze())
                with open(bn_path, 'w') as bottleneck_file:
                    bottleneck_file.write(bottleneck_string)

        except KeyError:
            tf.logging.error('Bottlenecks not created.')
        except OSError as e:
            print(e)
