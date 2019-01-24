from abc import ABC, abstractmethod
import os
from random import shuffle
import collections
import hashlib
import re
import numpy as np
import tensorflow as tf

from utils import cifar
from utils import data_utils
from ompy import fileio, ml

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

class AbstractDatasetLoader(ABC):
    def __init__(self, config):
        self.config = config
        self.iterator = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.training_init_op = None
        self.val_init_op = None
        self.test_init_op = None

    @abstractmethod
    def load_datasets(self, do_shuffle=True, train_repetitions=-1):
        pass

    def _get_iterator(self, dset):
        if not isinstance(self.iterator, tf.data.Iterator):
            self.iterator = tf.data.Iterator.from_structure(dset.output_types,
                                                            dset.output_shapes)

    def _create_iterators(self):

        if self.config.is_training.lower() == 'true':
            self._get_iterator(self.train_dataset)
            self.training_init_op = self.iterator.make_initializer(self.train_dataset)

        if int(self.config.validation_percentage) > 0:
            self._get_iterator(self.val_dataset)
            self.val_init_op = self.iterator.make_initializer(self.val_dataset)

        if int(self.config.testing_percentage) > 0:
            self._get_iterator(self.test_dataset)
            self.test_init_op = self.iterator.make_initializer(self.test_dataset)


    def initialize_train(self, sess):
        if self.config.is_training.lower() != 'true':
            raise RuntimeError('is_training flag is not set to true')
        sess.run(self.training_init_op)
    def initialize_val(self, sess):
        if int(self.config.validation_percentage) <= 0:
            raise RuntimeError('Validation set is set to 0%')
        sess.run(self.val_init_op)
    def initialize_test(self, sess):
        if int(self.config.testing_percentage) <= 0:
            raise RuntimeError('Test set is set to 0%')
        sess.run(self.test_init_op)

    def get_input(self):
        return self.iterator.get_next()
        
class TFRecordDatasetLoader(AbstractDatasetLoader):

    def __init__(self, config):
        super().__init__(config)
        self.num_batches = cifar.NUM_SAMPLES_TRAINING // self.config.batch_size

    def load_datasets(self, do_shuffle=True, train_repetitions=-1):
        try:
            if self.config.is_training.lower() == 'true':
                self.train_dataset = dset_from_tfrecord(cifar.get_data_path(self.config,'training'),
                                                        batch_size=self.config.batch_size,
                                                        do_shuffle=do_shuffle,
                                                        use_distortion=True,
                                                        repetitions=train_repetitions)
    
            if int(self.config.validation_percentage) > 0:
                self.val_dataset = dset_from_tfrecord(cifar.get_data_path(self.config,'validation'),
                                                        batch_size=self.config.batch_size,
                                                        repetitions=train_repetitions)
            if int(self.config.testing_percentage) > 0:
                self.test_dataset = dset_from_tfrecord(cifar.get_data_path(self.config,'testing'),
                                                        batch_size=self.config.batch_size,
                                                        repetitions=1)
            self._create_iterators()
        except IndexError as err:
            tf.logging.error(err.args)

class RegressionDatasetLoader(AbstractDatasetLoader):
    def __init__(self, config, process_images, process_maps):
        super().__init__(config)
        self.image_lists = self._create_file_lists()
        self.process_images = process_images
        self.process_maps = process_maps

        self.num_samples = len(self.image_lists['training'])
        self.num_batches = self.num_samples // self.config.batch_size

    def _create_file_lists(self, shuffle_all=True):
        """Builds train, val, and test sets from the file system.
        Returns:
          An OrderedDict containing an entry for each label subfolder, with images
          split into training, testing, and validation sets within each label.
          The order of items defines the class indices.
        """
        images_list = fileio.read_dir_to_list(self.config.data_path_samples)
        maps_list = fileio.read_dir_to_list(self.config.data_path_gt)
        assert( len(images_list) == len(maps_list))
        zipped_samples = list(zip(images_list, maps_list))

        if shuffle_all:
            shuffle(zipped_samples)

        return ml.create_subsets(zipped_samples,
                                 self.config.validation_percentage,
                                 self.config.testing_percentage)

    def load_datasets(self, do_shuffle=True, train_repetitions=-1):
        try:
            if self.config.is_training.lower() == 'true':
                self.train_dataset = dset_from_image_pair(self.image_lists['training'],
                                                      self.process_images,
                                                      self.process_maps,
                                                      batch_size=self.config.batch_size,
                                                      do_shuffle=do_shuffle,
                                                      repetitions=train_repetitions)

            if int(self.config.validation_percentage) > 0:
                self.val_dataset = dset_from_image_pair(self.image_lists['validation'],
                                                        self.process_images,
                                                        self.process_maps,
                                                        batch_size=self.config.batch_size,
                                                        do_shuffle=do_shuffle,
                                                        repetitions=train_repetitions)
             
            if int(self.config.testing_percentage) > 0:
                self.test_dataset = dset_from_image_pair(self.image_lists['testing'],
                                                         self.process_images,
                                                         self.process_maps,
                                                         batch_size=self.config.batch_size,
                                                         do_shuffle=do_shuffle,
                                                         repetitions=1)
            self._create_iterators()
        except IndexError as err:
            tf.logging.error(err.args)

class DatasetLoaderClassifier(AbstractDatasetLoader):
    def __init__(self, config):
        super().__init__(config)
        self.image_lists = self._create_file_lists()
        self.num_samples = None
        self.num_batches = None

    def load_datasets(self, process_func, do_shuffle=True, train_repetitions=-1):

        if self.config.is_training.lower() == 'true':
            samples_list, labels_list = get_files_from_ord_dict(self.image_lists,
                                                        self.config.data_path,
                                                        subset='training')
            self.num_samples = len(samples_list)
            self.num_batches = self.num_samples // self.config.batch_size

            self.train_dataset = dset_from_lists(samples_list,
                                             labels_list,
                                             process_func,
                                             batch_size=self.config.batch_size,
                                             do_shuffle=do_shuffle,
                                             repetitions=train_repetitions)

        if int(self.config.validation_percentage) > 0:
            samples_list, labels_list = get_files_from_ord_dict(self.image_lists,
                                                    self.config.data_path,
                                                    subset='validation')
            self.val_dataset = dset_from_lists(samples_list,
                                               labels_list,
                                               process_func,
                                               batch_size=self.config.batch_size,
                                               do_shuffle=do_shuffle,
                                               repetitions=train_repetitions)
         
        if int(self.config.testing_percentage) > 0:
            samples_list, labels_list = get_files_from_ord_dict(self.image_lists,
                                                    self.config.data_path,
                                                    subset='testing')
            self.test_dataset = dset_from_lists(samples_list,
                                                labels_list,
                                                process_func,
                                                batch_size=self.config.batch_size,
                                                do_shuffle=do_shuffle,
                                                repetitions=1)
        self._create_iterators()

    @abstractmethod
    def get_bottleneck_filenames():
        pass

class FileListDatasetLoader(DatasetLoaderClassifier):
    def __init__(self, config):
        super().__init__(config)

    def _create_file_lists(self):
        """Builds a list of training samples from a txt file.
      
        It parses the txt file. The txt file contains full paths to files and
        a class label.  Each line has exactly one (filename, label) pair,
        seperated by a space. After reading the files list, it splits them into stable
        training, testing, and validation sets, and returns a data structure
        describing the lists of images for each label and their paths.
      
        Returns:
          An OrderedDict containing an entry for each label subfolder, with images
          split into training, testing, and validation sets within each label.
          The order of items defines the class indices.
        """
        if not tf.gfile.Exists(self.config.data_path):
            tf.logging.error("File with samples '" + self.config.data_path + "' not found.")
            raise FileNotFoundError
        assert os.path.isfile(self.config.data_path)
        tf.logging.info("Looking for samples in '" + self.config.data_path + "'")
        file_list, label_list = fileio.parse_imgdb_list(self.config.data_path)
        
        try:
            class_names = __import__('configs.crowdnet_classes', fromlist=['class_names'])
            class_names = class_names.class_names
            #__import__(class_names, fromlist=[crowdnet_classes])
        except ImportError:
            tf.logging.warning('No class names have been found. Using a simple counter')
            class_names = list(np.unique(label_list))
        else:
            assert len(class_names) == len(np.unique(label_list))
    
        if not file_list:
            tf.logging.warning('No files found')
        if len(file_list) < 20:
            tf.logging.warning(
                'WARNING: Folder has less than 20 samples, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            tf.logging.warning(
                'WARNING: List {} has more than {} samples. Some samples will '
                'never be selected.'.format(self.config.data_path, MAX_NUM_IMAGES_PER_CLASS))
    
        result = collections.OrderedDict()
        for c in range(len(class_names)):
            result[c] = {
                'dir': str(class_names[c]),
                'training': [],
                'testing': [],
                'validation': [] }
    
        for i, file_name in enumerate(file_list):
            current_label = label_list[i]
            nt = (int(len(file_list)) *
                 (1.0 -
                  0.01 * (int(self.config.validation_percentage) +
                          int(self.config.testing_percentage))))

            nv = int(0.01 * len(file_list) * int(self.config.validation_percentage))
    
            if i < nt:
                result[current_label]['training'].append(file_name)
            elif i >= nt+nv:
                result[current_label]['testing'].append(file_name)
            else:
                result[current_label]['validation'].append(file_name)
    
        return result

    def get_bottleneck_filenames(self, root_dir, subset):
    
        def _get_long_file_name(sample, new_file_ext):
            file_name = os.path.basename(sample)
            prefix = os.path.basename(os.path.dirname(sample))
            long_file_name = prefix + '_' +  file_name
            long_file_name = os.path.splitext(long_file_name)[0] + new_file_ext
            return long_file_name

        samples_list = []
        labels_list = []
    
        for label, value in self.image_lists.items():
            samples_subset = self.image_lists[label][subset]
    
            os.makedirs(os.path.join(root_dir, self.image_lists[label]['dir']),
                        exist_ok=True)

            # Prepend full path to every file name in the subset
            # Replace image file extension with bottleneck file extension
            samples_subset = [os.path.join(root_dir,
                                           self.image_lists[label]['dir'],
                                           _get_long_file_name(sample, '.txt'))
                                 for sample in samples_subset]
    
            samples_list += samples_subset
    
        return samples_list

class DirectoryDatasetLoader(DatasetLoaderClassifier):
    def __init__(self, config):
        super().__init__(config)

    def _create_file_lists(self):
        """Builds a list of training samples from the file system.
      
        Analyzes the sub folders in the image directory, splits them into stable
        training, testing, and validation sets, and returns a data structure
        describing the lists of images for each label and their paths.
      
        Returns:
          An OrderedDict containing an entry for each label subfolder, with images
          split into training, testing, and validation sets within each label.
          The order of items defines the class indices.
        """
        if not tf.gfile.Exists(self.config.data_path):
            tf.logging.error("Samples root directory '" + self.config.data_path + "' not found.")
            raise FileNotFoundError
        assert os.path.isdir(self.config.data_path)
        result = collections.OrderedDict()
        sub_dirs = sorted(x[0] for x in tf.gfile.Walk(self.config.data_path))
        # The root directory comes first, so skip it.
        is_root_dir = True
        for sub_dir in sub_dirs:
            if is_root_dir:
                is_root_dir = False
                continue
            extensions = sorted(set(os.path.normcase(ext)  # Smash case on Windows.
                                  for ext in ['txt', 'TXT', 'PNG', 'png', 'JPG', 'jpg']))
            file_list = []
            dir_name = os.path.basename(sub_dir)
            if dir_name == self.config.data_path:
                continue
            tf.logging.info("Looking for samples in '" + dir_name + "'")
            for extension in extensions:
                file_glob = os.path.join(self.config.data_path, dir_name, '*.' + extension)
                file_list.extend(tf.gfile.Glob(file_glob))
            if not file_list:
                tf.logging.warning('No files found')
                continue
            if len(file_list) < 20:
                tf.logging.warning(
                    'WARNING: Folder has less than 20 samples, which may cause issues.')
            elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
                tf.logging.warning(
                    'WARNING: Folder {} has more than {} samples. Some samples will '
                    'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
            label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
            training_images = []
            testing_images = []
            validation_images = []
            for file_name in file_list:
                base_name = os.path.basename(file_name)
                # We want to ignore anything after '_nohash_' in the file name when
                # deciding which set to put an image in, the data set creator has a way of
                # grouping photos that are close variations of each other. For example
                # this is used in the plant disease data set to group multiple pictures of
                # the same leaf.
                hash_name = re.sub(r'_nohash_.*$', '', file_name)
                # This looks a bit magical, but we need to decide whether this file should
                # go into the training, testing, or validation sets, and we want to keep
                # existing files in the same set even if more files are subsequently
                # added.
                # To do that, we need a stable way of deciding based on just the file name
                # itself, so we do a hash of that and then use that to generate a
                # probability value that we use to assign it.
                hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(hash_name)).hexdigest()
                percentage_hash = ((int(hash_name_hashed, 16) %
                                    (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                                   (100.0 / MAX_NUM_IMAGES_PER_CLASS))
                if percentage_hash < int(self.config.validation_percentage):
                    validation_images.append(base_name)
                elif percentage_hash < (int(self.config.testing_percentage) +
                                        int(self.config.validation_percentage)):
                    testing_images.append(base_name)
                else:
                    training_images.append(base_name)
            result[label_name] = {
                'dir': dir_name,
                'training': training_images,
                'testing': testing_images,
                'validation': validation_images,
            }
        return result

    def get_bottleneck_filenames(self, root_dir, subset):
    
        samples_list = []
        labels_list = []
    
        for label, value in self.image_lists.items():
            samples_subset = self.image_lists[label][subset]
    
            os.makedirs(os.path.join(root_dir, self.image_lists[label]['dir']),
                        exist_ok=True)

            # Prepend full path to every file name in the subset
            # Replace image file extension with bottleneck file extension
            samples_subset = [os.path.join(root_dir,
                                           self.image_lists[label]['dir'],
                                           os.path.splitext(sample)[0] + '.txt')
                                 for sample in samples_subset]
    
            samples_list += samples_subset
    
        return samples_list

def get_files_from_ord_dict(ord_dict, root_dir, subset):

    samples_list = []
    labels_list = []
    label_number = 0

    for label, value in ord_dict.items():
        samples_subset = ord_dict[label][subset]

        #Prepend full path to every file name in the subset
        samples_subset = [os.path.join(root_dir, ord_dict[label]['dir'], sample)
                             for sample in samples_subset]

        samples_list += samples_subset
        labels_list += [label_number] * len(ord_dict[label][subset])
        label_number += 1

    return samples_list, labels_list

def dset_from_lists(samples_list,
                    labels_list,
                    process_func,
                    batch_size=64,
                    do_shuffle=True,
                    repetitions=-1):
    """Creates a TensorFlow Dataset instance from a list of files.
    Performs additional preprocessing using `process_func`.
    Returns:
        A `tf.data.dataset` containing pairs of (image, label)
    """
    if len(labels_list < 200):
        tf.logging.warning('Small dataset. Accidentally skipped any classes?')
    assert( len(labels_list) == len(samples_list))
    num_classes = len(np.unique(labels_list))

    # Create Dataset of images
    dset_x = tf.data.Dataset.from_tensor_slices(tf.constant(samples_list))
    dset_x = dset_x.map(process_func)

    # Create Dataset of maps
    dset_y = tf.data.Dataset.from_tensor_slices(tf.constant(labels_list))
    dset_y = dset_y.map(lambda z: tf.one_hot(z, num_classes))

    # Merge two datasets
    dset = tf.data.Dataset.zip((dset_x, dset_y))
    if do_shuffle:
        dset = dset.shuffle(4000)

    dset = dset.repeat(repetitions).batch(batch_size)
    return dset

def dset_from_image_pair(image_pairs,
                         process_images,
                         process_maps,
                         batch_size=64,
                         do_shuffle=True,
                         repetitions=-1):
    """ Creates a TensorFlow Dataset instance from pairs of image samples.
    Performs additional preprocessing. 
    Returns:
        A `tf.data.dataset` containing pairs of (image, map)
    """
    images_list = [p[0] for p in image_pairs]
    maps_list = [p[1] for p in image_pairs]
    
    if len(images_list) != len(maps_list) or len(maps_list) == 0:
        raise IndexError('Wrong # of samples in dataset',
                        len(images_list),len(maps_list))

    # Create Dataset of images
    dset_x = tf.data.Dataset.from_tensor_slices(tf.constant(images_list))
    dset_x = dset_x.map(process_images)

    # Create Dataset of maps
    dset_y = tf.data.Dataset.from_tensor_slices(tf.constant(maps_list))
    # tf.py_func makes usage of external libaries possible.
    dset_y = dset_y.map(
        lambda filename: tf.py_func( process_maps, [filename], tf.float32))

    def _enforce_shape(map_tensor):
        map_tensor.set_shape([56,56,1])
        return map_tensor

    dset_y = dset_y.map(_enforce_shape)

    # Merge two datasets
    dset = tf.data.Dataset.zip((dset_x, dset_y))
    if do_shuffle:
        dset = dset.shuffle(4000)

    dset = dset.repeat(repetitions).batch(batch_size)
    return dset

def dset_from_tfrecord(tfrecord_file,
                       max_samples=-1, 
                       batch_size=64,
                       do_shuffle=False,
                       use_distortion=False,
                       repetitions=-1):

    dset = tf.data.TFRecordDataset(tfrecord_file).repeat(repetitions)

    #TODO
    dset = dset.map(data_utils.parse_tf_example, num_parallel_calls=batch_size)
    if use_distortion:
        dset = dset.map(data_utils.distort_image, num_parallel_calls=batch_size)

    #TODO Only during training? Best buffer size?
    if do_shuffle:
        min_queue_examples = int(cifar.NUM_SAMPLES_TRAINING * 0.4)
        # Ensure that the capacity is sufficiently large to provide good random
        # shuffling.
        dset = dset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

    dset = dset.batch(batch_size)
        
    # TODO Only take up to `max_samples` samples from the data.
    #dset = dset.take(max_samples)

    return dset

def get_IMGDB_dataset(img_list_filename,
                        subtract_mean=False,
                        normalize_data=False):

    if subtract_mean or normalize_data:
        raise NotImplementedError

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
