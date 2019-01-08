from __future__ import print_function

from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import tensorflow as tf
import os
from imageio import imread
from skimage.transform import resize
from skimage import filters, util
import platform

from ompy import fileio
from ompy import ml

def load_image_and_blur(filename):
    image = imread(filename.decode())
    image = util.img_as_float32(image)
    image = filters.gaussian(image) # Blur image
    image = resize(image, [28, 28], mode='reflect')
    image = image.astype('float32')
    image = image[:,:,np.newaxis] # TF insists on 3rd dimension.
    return image

def parse_txt(filename):
    raw_string = tf.read_file(filename)
    #TODO warn if values do not return constant number of bottleneck features
    raw_values = tf.string_split([raw_string], delimiter=',').values
    float_values = tf.strings.to_number(raw_values, out_type=tf.float32)
    float_values.set_shape([25088])
    return float_value

def parse_png(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [448, 448])
    return image_resized

def float_string_to_list(filename):
    """Converts a ascii list of float numbers to a python list of floats

    Expects a filename to a file which holds one line of
    comma-seperated values like
    0.0, 0.86999977, ... 0.0, 0.0, 5.131418
    It converts it to a python list of float values.
    Args:
        filename: A `string`, the full path to the file.
    Returns:
        A Python `list` of float values from the specified file.
    Raises:
        `ValueError` if the values are invalid
        `FileNotFoundError` if filename is not a file.
    """
    try:
        with open(filename, 'r') as the_file:
            bottleneck_string = the_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except ValueError:
        print('Invalid float found')
    except FileNotFoundError:
        print('Cannot find file: %s'.format(bottleneck_path))
    return bottleneck_values

def print_shape(data):
    try:
        d = iter(data)
    except TypeError:
        print('This is just one element. Shape: {}.'.format(data.shape))
    else:
        print('Dataset size:')
        for d in data:
            print(d.shape)

def get_random_data():
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))

    return data, labels

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float32")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))

    # Cast elements of labels vectors to integers
    Ytr = Ytr.astype(int)
    Yte = Yte.astype(int)
    return Xtr, Ytr, Xte, Yte

def load_IMGDB(txt_list, img_shape=(32,32,3)):
    """
    Load images from the IMGDB image list in txt_list and put them into
    numpy arrays.
    """
    image_names, image_labels = fileio.parse_imgdb_list(txt_list=txt_list)
    assert len(image_names) == len(image_labels)
    num_images = len(image_names)

    image_labels = np.asarray(image_labels, dtype=np.int32)
    image_tensor = np.empty([num_images, *img_shape], dtype=np.float32)

    for i in range(num_images):
        try:
            img = imread(image_names[i], pilmode='RGB')
            img = resize(img, img_shape)
            image_tensor[i] = img
        except ValueError as error:
            print('{}: {}'.format(i,error))
            image_tensor = np.delete(image_tensor, i, axis=0)
            image_labels = np.delete(image_labels, i)
        except IndexError as error:
            print(error)
            if i > 1:
                image_tensor = image_tensor[0:i-1]
                image_labels = image_labels[0:i-1]
            break

    return image_tensor, image_labels

def get_some_data(input_path, 
                  input_path_imgdb_test=None,
                  dataset_name=None,
                  img_shape=(32, 32, 3),
                  subtract_mean=True,
                  normalize_data=False,
                  channels_first=True,
                  flatten_imgs=True,
                  one_hot=False):
    """
    Load the CIFAR-10 or IMGDB dataset from disk and perform preprocessing to prepare
    it for classifiers. 
    """
    X_train= X_val= X_test = []
    y_train= y_val= y_test = []

    # Load the CIFAR-10 dataset
    if dataset_name == 'cifar':

        num_training=49000
        num_validation=1000
        num_test=10000

        X_train, y_train, X_test, y_test = load_CIFAR10(input_path)

        # Subsample the data
        mask = list(range(num_training, num_training + num_validation))
        X_val = X_train[mask]
        y_val = y_train[mask]
        mask = list(range(num_training))
        X_train = X_train[mask]
        y_train = y_train[mask]
        mask = list(range(num_test))
        X_test = X_test[mask]
        y_test = y_test[mask]

    # Load the IMGDB dataset
    elif dataset_name == 'imgdb':
        X, y = load_IMGDB(input_path, img_shape=img_shape)
        nt = int(X.shape[0] * 0.8)
        nv = int(X.shape[0] * 0.1)
        X_train = X[:nt]
        y_train = y[:nt]
        X_val = X[nt:nt+nv]
        y_val = y[nt:nt+nv]
        X_test = X[nt+nv:]
        y_test = y[nt+nv:]
    else:
        raise NotImplementedError


    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    if normalize_data:
        X_train = X_train / 255.0
        X_val = X_val / 255.0
        X_test = X_test / 255.0

    # Transpose so that channels come first
    if channels_first:
        X_train = X_train.transpose(0, 3, 1, 2).copy()
        X_val = X_val.transpose(0, 3, 1, 2).copy()
        X_test = X_test.transpose(0, 3, 1, 2).copy()

    # For a fully-connected net, reshape each samples to a single rows.
    if flatten_imgs:
        X_train = np.reshape(X_train,[X_train.shape[0], -1])
        X_val = np.reshape(X_val,[X_val.shape[0], -1])
        X_test = np.reshape(X_test,[X_test.shape[0], -1])
    
    # One-hot encode the labels
    if one_hot:
        y_train = ml.makeonehot(y_train)
        y_val = ml.makeonehot(y_val)
        y_test = ml.makeonehot(y_test)

    return (X_train, y_train, X_val, y_val, X_test, y_test)


def load_tiny_imagenet(path, dtype=np.float32, subtract_mean=True):
    """
    Load TinyImageNet. Each of TinyImageNet-100-A, TinyImageNet-100-B, and
    TinyImageNet-200 have the same directory structure, so this can be used
    to load any of them.

    Inputs:
    - path: String giving path to the directory to load.
    - dtype: numpy datatype used to load the data.
    - subtract_mean: Whether to subtract the mean training image.

    Returns: A dictionary with the following entries:
    - class_names: A list where class_names[i] is a list of strings giving the
      WordNet names for class i in the loaded dataset.
    - X_train: (N_tr, 3, 64, 64) array of training images
    - y_train: (N_tr,) array of training labels
    - X_val: (N_val, 3, 64, 64) array of validation images
    - y_val: (N_val,) array of validation labels
    - X_test: (N_test, 3, 64, 64) array of testing images.
    - y_test: (N_test,) array of test labels; if test labels are not available
      (such as in student code) then y_test will be None.
    - mean_image: (3, 64, 64) array giving mean training image
    """
    # First load wnids
    with open(os.path.join(path, 'wnids.txt'), 'r') as f:
        wnids = [x.strip() for x in f]

    # Map wnids to integer labels
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    # Use words.txt to get names for each class
    with open(os.path.join(path, 'words.txt'), 'r') as f:
        wnid_to_words = dict(line.split('\t') for line in f)
        for wnid, words in wnid_to_words.items():
            wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
    class_names = [wnid_to_words[wnid] for wnid in wnids]

    # Next load training data.
    X_train = []
    y_train = []
    for i, wnid in enumerate(wnids):
        if (i + 1) % 20 == 0:
            print('loading training data for synset %d / %d'
                  % (i + 1, len(wnids)))
        # To figure out the filenames we need to open the boxes file
        boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
        with open(boxes_file, 'r') as f:
            filenames = [x.split('\t')[0] for x in f]
        num_images = len(filenames)

        X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)
        y_train_block = wnid_to_label[wnid] * \
                        np.ones(num_images, dtype=np.int64)
        for j, img_file in enumerate(filenames):
            img_file = os.path.join(path, 'train', wnid, 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
        ## grayscale file
                img.shape = (64, 64, 1)
            X_train_block[j] = img.transpose(2, 0, 1)
        X_train.append(X_train_block)
        y_train.append(y_train_block)

    # We need to concatenate all training data
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Next load validation data
    with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
        img_files = []
        val_wnids = []
        for line in f:
            img_file, wnid = line.split('\t')[:2]
            img_files.append(img_file)
            val_wnids.append(wnid)
        num_val = len(img_files)
        y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
        X_val = np.zeros((num_val, 3, 64, 64), dtype=dtype)
        for i, img_file in enumerate(img_files):
            img_file = os.path.join(path, 'val', 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
                img.shape = (64, 64, 1)
            X_val[i] = img.transpose(2, 0, 1)

    # Next load test images
    # Students won't have test labels, so we need to iterate over files in the
    # images directory.
    img_files = os.listdir(os.path.join(path, 'test', 'images'))
    X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)
    for i, img_file in enumerate(img_files):
        img_file = os.path.join(path, 'test', 'images', img_file)
        img = imread(img_file)
        if img.ndim == 2:
            img.shape = (64, 64, 1)
        X_test[i] = img.transpose(2, 0, 1)

    y_test = None
    y_test_file = os.path.join(path, 'test', 'test_annotations.txt')
    if os.path.isfile(y_test_file):
        with open(y_test_file, 'r') as f:
            img_file_to_wnid = {}
            for line in f:
                line = line.split('\t')
                img_file_to_wnid[line[0]] = line[1]
        y_test = [wnid_to_label[img_file_to_wnid[img_file]]
                  for img_file in img_files]
        y_test = np.array(y_test)

    mean_image = X_train.mean(axis=0)
    if subtract_mean:
        X_train -= mean_image[None]
        X_val -= mean_image[None]
        X_test -= mean_image[None]

    return {
      'class_names': class_names,
      'X_train': X_train,
      'y_train': y_train,
      'X_val': X_val,
      'y_val': y_val,
      'X_test': X_test,
      'y_test': y_test,
      'class_names': class_names,
      'mean_image': mean_image,
    }


def load_models(models_dir):
    """
    Load saved models from disk. This will attempt to unpickle all files in a
    directory; any files that give errors on unpickling (such as README.txt)
    will be skipped.

    Inputs:
    - models_dir: String giving the path to a directory containing model files.
      Each model file is a pickled dictionary with a 'model' field.

    Returns:
    A dictionary mapping model file names to models.
    """
    models = {}
    for model_file in os.listdir(models_dir):
        with open(os.path.join(models_dir, model_file), 'rb') as f:
            try:
                models[model_file] = load_pickle(f)['model']
            except pickle.UnpicklingError:
                continue
    return models


def load_imagenet_val(num=None):
    """Load a handful of validation images from ImageNet.

    Inputs:
    - num: Number of images to load (max of 25)

    Returns:
    - X: numpy array with shape [num, 224, 224, 3]
    - y: numpy array of integer image labels, shape [num]
    - class_names: dict mapping integer label to class name
    """
    imagenet_fn = 'cs231n/datasets/imagenet_val_25.npz'
    if not os.path.isfile(imagenet_fn):
      print('file %s not found' % imagenet_fn)
      print('Run the following:')
      print('cd cs231n/datasets')
      print('bash get_imagenet_val.sh')
      assert False, 'Need to download imagenet_val_25.npz'
    f = np.load(imagenet_fn)
    X = f['X']
    y = f['y']
    class_names = f['label_map'].item()
    if num is not None:
        X = X[:num]
        y = y[:num]
    return X, y, class_names
