import sys
import os

sys.path.extend(['..'])
import tensorflow as tf
import numpy as np
from imageio import imread
from skimage.transform import resize
from utils import datahandler
from tqdm import tqdm
from abc import ABC, abstractmethod

from tensorflow.python import debug as tf_debug

def save_batch(data_batch, path_name):
    """ Save data batch to npy files.
    """
    if not isinstance(data_batch, np.ndarray):
        raise ValueError('Expects a numpy array')
    os.makedirs(path_name, exist_ok=True)
    for i in range(data_batch.shape[0]):
        np.save(os.path.join(path_name,'pred_{}.npy'.format(i)), data_batch[i])

class Trainer(ABC):
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

    @abstractmethod
    def _get_monitor_ops(self, extra_ops=[]):
        pass

    @abstractmethod
    def _run_graph(self, fetches):
        pass

    @abstractmethod
    def _keep_printable_keys(d):
        """Remove all non-printable dict entrys, and keep the rest"""
        pass

    @abstractmethod
    def _test_loop(self):
        pass

    def train(self):
        try:
            if self.data_loader == None or self.config.is_training.lower() != 'true':
                raise RuntimeError('No training data or `\"is_training\"!= \"True\"`?!')
    
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
            for i in range(self.config.num_epochs):
                tf.logging.info('train(): ===== EPOCH {} ====='.format(i))
                # Initialize iterator with training data
                self.data_loader.initialize_train(self.sess)
                tf.logging.info('Initial loss: {}'.format(self.sess.run(self.model.loss)))
                #Do not monitor, just train for one epoch
                try:
                    for _ in tqdm(range(self.data_loader.num_batches), ascii=True, desc='epoch'):
                        self.sess.run([self.model.optimize])
                except tf.errors.OutOfRangeError as err:
                    tf.logging.warning(err.args)
        
                # Monitor the training after every epoch
                fetches = self._get_monitor_ops(extra_ops=[global_step])
                train_output = self._run_graph(fetches)
                train_writer.add_summary(train_output['summary'],
                                         global_step=train_output['global_step'])
                train_writer.flush()
    
                #Check how it goes with the validation set
                self.data_loader.initialize_val(self.sess)
                fetches = self._get_monitor_ops(extra_ops=[global_step])
                val_output = self._run_graph(fetches)
    
                val_writer.add_summary(val_output['summary'],
                                         global_step=val_output['global_step'])
                val_writer.flush()

                Trainer._keep_printable_keys(train_output)
                Trainer._keep_printable_keys(val_output)
    
                save_path = saver.save(self.sess, self.config.checkpoint_dir + 'run_' + str(train_id))
                tf.logging.info('train(): Model checkpoint saved to %s' % save_path)
        
            train_writer.close()
            val_writer.close()

        except RuntimeError as err:
            tf.logging.error(err.args)
            try:
                train_writer
            except NameError:
                pass
            else:
                train_writer.close()
            try:
                val_writer
            except NameError:
                pass
            else:
                val_writer.close()



    def test(self, checkpoint_dir=None):

        try:
            # Load parameters
            if checkpoint_dir != None:
                saver = tf.train.Saver()
                saver.restore(self.sess, checkpoint_dir)

            # Check if test data is loaded
            if self.data_loader == None or int(self.config.testing_percentage) <= 0:
                raise RuntimeError('No test data or testset set to 0%! Check JSON config')
            # Load test dataset
            self.data_loader.initialize_test(self.sess)
            self._test_loop()
        except RuntimeError as err:
            tf.logging.error(err.args)

    def predict(self, image_path, num_images=1):
        try:
            img1 = imread(image_path, format='RGB')
            img1 = resize(img1, (224, 224))

            prob = self.sess.run(self.model.softmax,
                                    feed_dict={self.model.data: [img1]})[0]
            preds = (np.argsort(prob)[::-1])[0:5]
            for p in preds:
                print(class_names[p], prob[p])

        except (FileNotFoundError, OSError) as err:
            tf.logging.error(err.args)

    def create_bottlenecks(self, subset):
        try:
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

        except KeyError as err:
            tf.logging.error(err.args)
            tf.logging.error('Check if bottlenecks have been created.')
        except (OSError, RuntimeError, NotImplementedError) as err:
            tf.logging.error(err.args)

class RegressionTrainer(Trainer):

    def _get_monitor_ops(self, extra_ops=[]):
        fetches = [self.model.loss,
                   self.model.mae,
                   self.write_op]
        fetches.extend(extra_ops)
        return fetches

    def _run_graph(self, fetches):
        loss, mae, summary, global_step = self.sess.run(fetches)

        return {'loss':loss,
                'mae':mae,
                'summary':summary,
                'global_step':global_step}

    def _test_loop(self):
        maes = []
        try:
            bn = 0
            while True:
                mae, prediction = self.sess.run([self.model.mae, self.model.prediction])
                tf.logging.info('Per batch Mean Absolute Error: {}'.format(mae))
                maes.append(mae)
                save_batch(
                    prediction, 
                    os.path.join(self.config.data_path_pred,'batch_{}'.format(bn)))
                bn += 1
        except tf.errors.OutOfRangeError:
            pass
        tf.logging.info('Average MAE of batch MAE: {} (std: {})'.format(
                            np.mean(maes),
                            np.std(maes)))

    @staticmethod
    def _keep_printable_keys(d):
        try:
            d.pop('summary')
            d.pop('global_step')
        except KeyError as err:
            tf.logging.error(err.args)


class ClassificationTrainer(Trainer):

    def _get_monitor_ops(self, extra_ops=[]):
        fetches = [self.model.loss,
                   self.model.accuracy,
                   self.model.cm,
                   self.model.softmax,
                   self.write_op]
        fetches.extend(extra_ops)
        return fetches

    def _run_graph(self, fetches):
        loss, accuracy, cm, softmx, summary, global_step = self.sess.run(fetches)

        return {'loss':loss,
                'accuracy':accuracy,
                'cm':cm,
                'softmax':softmx,
                'summary':summary,
                'global_step':global_step}

    def _test_loop(self):
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

    @staticmethod
    def _keep_printable_keys(d):
        try:
            d.pop('summary')
            d.pop('softmax')
            d.pop('global_step')
        except KeyError as err:
            tf.logging.error(err.args)

