"""
This class will contain different loaders for cifar 100 dataset
# Techniques
# FIT in ram
# - load numpys in the graph - Cifar100DataLoaderNumpy
# - generator python - BaselineCifar100Loader

# Doesn't fit in ram
# - load files but in tfrecords format - Cifar100TFRecord
# - load files from disk using dataset api - Cifar100IMGLoader

Supports IMGDB4 and CIFAR dataset
"""
from utils import data_utils, dirs
from ompy import fileio

from abc import ABC, abstractmethod
import collections
import os
import hashlib
import re
import tensorflow as tf
import numpy as np

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

class ImgdbLoader:
    """
    Loads IMGDB dataset using TensorFlow-Dataset API
    """
    def __init__(self,config):
        print('Loading data...')
        self.config = config
        # Load data as numpy array
        data = data_utils.get_some_data(
            self.config.input_path,
            dataset_name=self.config.dataset_name,
            img_shape=(224, 224, 3),
            normalize_data=False,
            subtract_mean=False,
            channels_first=False,
            flatten_imgs=False)

        data_utils.print_shape(data)
        X_train, y_train, X_val, y_val, X_test, y_test = data
#        #DEBUG - small subset to provoke overfitting
#        idx_overfit=np.random.choice(len(X_train),size=256,replace=False)
#        X_train= X_train[idx_overfit]
#        y_train= y_train[idx_overfit]
        self.num_batches = len(X_train) // self.config.batch_size

        self.train_dataset = self._from_numpy(X_train, y_train)
        self.val_dataset = self._from_numpy(X_val, y_val)
        self.test_dataset = self._from_numpy(X_test, y_test)

        self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types,
                                                   self.train_dataset.output_shapes)

        self.training_init_op = self.iterator.make_initializer(self.train_dataset)
        self.val_init_op = self.iterator.make_initializer(self.val_dataset)
        self.test_init_op = self.iterator.make_initializer(self.test_dataset)

    def _from_numpy(self, X, y):
        #TODO Improve using this guide:
        # https://www.tensorflow.org/guide/datasets#consuming_numpy_arrays
        num_classes = len(np.unique(y))
        dset_x = tf.data.Dataset.from_tensor_slices(X)
        dset_y = tf.data.Dataset.from_tensor_slices(y).map(
            lambda z: tf.one_hot(z, num_classes))
        dset = tf.data.Dataset.zip((dset_x, dset_y))
        dset = dset.shuffle(buffer_size=len(X))
        dset = dset.repeat(1).batch(self.config.batch_size)
        return dset


    def initialize_train(self, sess):
        sess.run(self.training_init_op)
    def initialize_val(self, sess):
        sess.run(self.val_init_op)
    def initialize_test(self, sess):
        sess.run(self.test_init_op)
    def get_input(self):
        return self.iterator.get_next()

class AbstractDatasetLoader(ABC):
    def __init__(self, config):

        self.config = config

        self.image_lists = self.create_file_lists()

    def load_datasets(self,do_shuffle=True, is_png=True, train_repetitions=-1):

        self.train_dataset, self.num_samples = dset_from_ordered_dict(
                                                    self.image_lists,
                                                    self.config.input_path,
                                                    subset='training',
                                                    batch_size=self.config.batch_size,
                                                    do_shuffle=do_shuffle,
                                                    is_png=is_png,
                                                    repetitions=train_repetitions)

        if int(self.config.validation_percentage) > 0:
            self.val_dataset, _ = dset_from_ordered_dict(
                                      self.image_lists, 
                                      self.config.input_path,
                                      subset='validation',
                                      batch_size=self.config.batch_size,
                                      do_shuffle=do_shuffle,
                                      is_png=is_png,
                                      repetitions=train_repetitions)

        if int(self.config.testing_percentage) > 0:
            self.test_dataset, _ = dset_from_ordered_dict(
                                       self.image_lists, 
                                       self.config.input_path,
                                       subset='testing',
                                       batch_size=self.config.batch_size,
                                       do_shuffle=do_shuffle,
                                       is_png=is_png,
                                       repetitions=1)

        self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types,
                                                        self.train_dataset.output_shapes)

        self.training_init_op = self.iterator.make_initializer(self.train_dataset)

        if int(self.config.validation_percentage) > 0:
            self.val_init_op = self.iterator.make_initializer(self.val_dataset)
        if int(self.config.testing_percentage) > 0:
            self.test_init_op = self.iterator.make_initializer(self.test_dataset)

        self.num_batches = self.num_samples // self.config.batch_size

    def initialize_train(self, sess):
        sess.run(self.training_init_op)
    def initialize_val(self, sess):
        sess.run(self.val_init_op)
    def initialize_test(self, sess):
        sess.run(self.test_init_op)
    def get_input(self):
        return self.iterator.get_next()

    @abstractmethod
    def create_file_lists():
        pass

    @abstractmethod
    def get_bottleneck_filenames():
        pass

class FileListDatasetLoader(AbstractDatasetLoader):

    def create_file_lists(self):
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
        if not tf.gfile.Exists(self.config.input_path):
            tf.logging.error("File with samples '" + self.config.input_path + "' not found.")
            raise FileNotFoundError
        assert os.path.isfile(self.config.input_path)
        tf.logging.info("Looking for samples in '" + self.config.input_path + "'")
        file_list, label_list = fileio.parse_imgdb_list(self.config.input_path)
        
        try:
            class_names = __import__('utils.crowdnet_classes', fromlist=['class_names'])
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
                'never be selected.'.format(self.config.input_path, MAX_NUM_IMAGES_PER_CLASS))
    
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

class DirectoryDatasetLoader(AbstractDatasetLoader):

    def create_file_lists(self):
        """Builds a list of training samples from the file system.
      
        Analyzes the sub folders in the image directory, splits them into stable
        training, testing, and validation sets, and returns a data structure
        describing the lists of images for each label and their paths.
      
        Returns:
          An OrderedDict containing an entry for each label subfolder, with images
          split into training, testing, and validation sets within each label.
          The order of items defines the class indices.
        """
        if not tf.gfile.Exists(self.config.input_path):
            tf.logging.error("Samples root directory '" + self.config.input_path + "' not found.")
            raise FileNotFoundError
        assert os.path.isdir(self.config.input_path)
        result = collections.OrderedDict()
        sub_dirs = sorted(x[0] for x in tf.gfile.Walk(self.config.input_path))
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
            if dir_name == self.config.input_path:
                continue
            tf.logging.info("Looking for samples in '" + dir_name + "'")
            for extension in extensions:
                file_glob = os.path.join(self.config.input_path, dir_name, '*.' + extension)
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

def dset_from_ordered_dict(ord_dict,
                           root_dir,
                           subset,
                           batch_size=64,
                           do_shuffle=True,
                           repetitions=-1,
                           is_png=True):

    num_classes = len(ord_dict.keys())

    def _parse_txt(filename):

        raw_string = tf.read_file(filename)
        #TODO warn if values do not return constant number of bottleneck features
        raw_values = tf.string_split([raw_string], delimiter=',').values
        float_values = tf.strings.to_number(raw_values, out_type=tf.float32)
        float_values.set_shape([2048])
        return float_values

    def _parse_png(filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string, channels=3)
        image_resized = tf.image.resize_images(image_decoded, [64, 64])
        return image_resized

    samples_list, labels_list = get_files_from_ord_dict(ord_dict, root_dir, subset)
    dset_x = tf.data.Dataset.from_tensor_slices(tf.constant(samples_list))
    if is_png:
        dset_x = dset_x.map(_parse_png)
    else:
        dset_x = dset_x.map(_parse_txt)
    dset_y = tf.data.Dataset.from_tensor_slices(tf.constant(labels_list))
    dset_y = dset_y.map(lambda z: tf.one_hot(z, num_classes))

    dset = tf.data.Dataset.zip((dset_x, dset_y))
    if do_shuffle:
        #dset = dset.shuffle()
        pass

    dset = dset.repeat(repetitions).batch(batch_size)
    num_samples = len(samples_list)
    return dset, num_samples


  
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
