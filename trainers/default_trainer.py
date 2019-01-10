import sys
import os

sys.path.extend(['..'])
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from utils import datahandler
from tqdm import tqdm

from tensorflow.python import debug as tf_debug

def save_batch(data_batch, path_name):
    """ Save data batch to npy files.
    """
    if not isinstance(data_batch, np.ndarray):
        raise ValueError('Expects a numpy array')
    os.makedirs(path_name, exist_ok=True)
    for i in range(data_batch.shape[0]):
        np.save(os.path.join(path_name,'pred_{}.npy'.format(i)), data_batch[i])

class Trainer:
    def __init__(self, sess, model, config, data_loader=None):
        """
        Constructing the trainer
        :param sess: tf.Session() instance
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
        self.write_op = tf.summary.merge_all()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def train(self):
        if self.data_loader == None:
            raise RuntimeError('No data loaded')

        train_id = np.random.randint(1e6,size=1)[0]
        tf.logging.info('train(): train_id: {}'.format(train_id))
        tf.logging.info('train(): Initializing data...')
        
        global_step = tf.train.get_or_create_global_step()
    
        saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        # File writers for TensorBoard
        train_writer = tf.summary.FileWriter( 
            self.config.summary_dir + 'run_' + str(train_id) + '/train', self.sess.graph)
        val_writer = tf.summary.FileWriter( 
            self.config.summary_dir + 'run_' + str(train_id) + '/val', self.sess.graph)
    
        tf.logging.info('train(): Training for {} epochs...'.format(self.config.num_epochs))
        #self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        for i in range(self.config.num_epochs):
            tf.logging.info('train(): ===== EPOCH {} ====='.format(i))
            # Initialize iterator with training data
            self.data_loader.initialize_train(self.sess)
            tf.logging.info('Initial loss: {}'.format(self.sess.run(self.model.loss)))
            #Do not monitor, just train for one epoch
            for _ in tqdm(range(self.data_loader.num_batches), ascii=True, desc='epoch'):
                self.sess.run([self.model.optimize])
    
            # Monitor the training after every epoch
            fetches = [self.model.optimize,
                       self.model.loss,
                       self.model.mae,
                       self.write_op,
                       global_step]

            _,train_loss, train_mae, summary_train, global_step_vl = self.sess.run(fetches)
            train_writer.add_summary(summary_train, global_step=global_step_vl)
            train_writer.flush()

            #Check how it goes with the validation set
            self.data_loader.initialize_val(self.sess)
            fetches_val = [self.model.loss,
                           self.model.mae,
                           self.write_op,
                           global_step]

            val_loss, val_mae, summary_val, global_step_vl = self.sess.run(fetches_val)
            tf.logging.info(('train(): #{}: train_loss: {} train_mae: {}' 
                                 ' val_loss: {} val_mae: {}').format(
                                     global_step_vl,
                                     train_loss, train_mae,
                                     val_loss, val_mae))

            val_writer.add_summary(summary_val, global_step=global_step_vl)
            val_writer.flush()

            save_path = saver.save(self.sess, self.config.checkpoint_dir + 'run_' + str(train_id))
            tf.logging.info('train(): Model checkpoint saved to %s' % save_path)
    
        train_writer.close()
        val_writer.close()

    def test(self, checkpoint_dir=None):

        # Load parameters
        if checkpoint_dir != None:
            saver = tf.train.Saver()
            saver.restore(self.sess, checkpoint_dir)

        # Check if test data is loaded
        if self.data_loader == None or int(self.config.testing_percentage) <= 0:
            raise RuntimeError('No test data available!')
        # Load test dataset
        self.data_loader.initialize_test(self.sess)
        maes = []
        #num_classes = int(self.data_loader.test_dataset.output_shapes[1][1])
        #confusion_matrix = np.zeros((num_classes, num_classes),dtype=int)
        try:
            while True:
                mae, prediction = self.sess.run([self.model.mae, self.model.prediction])
                tf.logging.info('Per batch Mean Absolute Error: {}'.format(mae))
                maes.append(mae)
                save_batch(prediction, self.config.data_path_pred)
        except tf.errors.OutOfRangeError:
            pass
        #accuracies = np.asarray(accuracies)
        tf.logging.info('Average MAE of batch MAE: {} (std: {})'.format(
                            np.mean(maes),
                            np.std(maes)))

    def predict(self, image_path, num_images=1):
        if num_images == 1:
            img1 = imread(image_path, mode='RGB')
            img1 = imresize(img1, (224, 224))
        else:
            raise NotImplementedError('Only the prediction of exactly one image is supported.')

        prob = self.sess.run(self.model.softmax,
                                feed_dict={self.model.data: [img1]})[0]
        preds = (np.argsort(prob)[::-1])[0:5]
        for p in preds:
            print(class_names[p], prob[p])

    def create_bottlenecks(self, subset):
        if self.data_loader == None:
            raise RuntimeError('No data loaded')
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
            raise NotImplementedError('subset must be either \'training\' or \'validation\'')

        try:
            bottleneck_paths = self.data_loader.get_bottleneck_filenames(
                                                    self.config.bottleneck_dir,   
                                                    subset=subset)

            for bn_path in tqdm(bottleneck_paths,ascii=True, desc='bottlenecks'):
                # Specify bottleneck layer here:
                fetches = [self.model.bottlenecks]
                #Get bottleneck feature vector for an image
                bottleneck_values = self.sess.run(fetches)[0]
                bottleneck_string = ','.join(str(x) for x in bottleneck_values.squeeze())
                with open(bn_path, 'w') as bottleneck_file:
                    bottleneck_file.write(bottleneck_string)

        except KeyError:
            tf.logging.error('Bottlenecks not created.')
        except OSError as e:
            print(e)
