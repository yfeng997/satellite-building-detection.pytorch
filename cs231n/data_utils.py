from __future__ import print_function

from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
import platform
from tqdm import tqdm_notebook as tqdm
import simplejson as json
import cv2
import re
import math
import sys

def load_fmow(param, subtract_mean=True, load_train=True, load_test=True):
    """
    Load Functional Map of World. Each of fMoW training set and validation
    set has the same directory structure, so this could be used to load any
    of them. 
    
    Inputs:
    - path: String giving path to the directory to load
      path structure: 
      /train
        /airport
          /airport0
            /airport0_0_rgb.jpg
            /airport0_0_rgb.json
            /airport0_0_msrgb.jpg
            /airport0_0_msrgb.json
            /airport0_1_rgb.jpg
            /airport0_1_rgb.json
            /airport0_1_msrgb.jpg
            /airport0_1_msrgb.json
          /airport1
        /barn
          /barn0
          /barn1
        /single_residential
      /val
        /airport
        /barn
        /single_residential
      /test
        /0001
          /0001_0_rgb.jpg
          /0001_0_rgb.json
          /0001_1_rgb.jpg
          /0001_1_rgb.json
        /0002
    - subtract_mean: boolean indicating whether to subtract the mean training image
    
    Returns: A dictionary with the following entries
    - X_train: (N_tr, 3, 200, 200) array of training images
    - y_train: (N_tr, ) array of training labels
    - X_val: (N_val, 3, 200, 200) array of validation images
    - y_val: (N_val, ) array of validation labels
    - X_test: (N_test, 3, 200, 200) array of test images
    - y_test: (N_test, ) array of test labels; if not available, then 
      y_test will be None
    - mean_image: (3, 200, 200) giving mean training image
    """
        
    # Load training data, validation data and test data at once
    X_train = np.array([], dtype=np.float32).reshape(0, 1, 224, 224)
    y_train = np.array([], dtype=np.int32).reshape(0, )
    X_test = np.array([], dtype=np.float32).reshape(0, 1, 224, 224)
    y_test = np.array([], dtype=np.int32).reshape(0, )
    
    counts = {}
    if load_train:
        print("Extracting Training and Validation Images...")
        for category in tqdm(params['fmow_class_names']):
            print("Extracting images from %s..." % category)
            path = os.path.join(params['dataset'], 'train', category)
            counts[cat_name] = sys.maxsize
            X_train, y_train = _process_dir(params, path, category, X_train, y_train, counts)
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.int32)
        X_val = X_train[X_train.shape[0]*0.9:, :, :, :]
        y_val = y_train[X_train.shape[0]*0.9:, ]
        X_train = X_train[:X_train.shape[0]*0.9, :, :, :]
        y_train = y_train[:X_train.shape[0]*0.9, ]
        mean_image = np.mean(X_train, axis=0)
    
    if load_test:
        print("Extracting Test Images...")
        counts[""] = sys.maxsize
        path = os.path.join(params['dataset'], 'test')
        X_test, y_test = _process_fir(params, path, "", X_test, y_test, counts)  
        X_test = X_test.astype(np.float32)
        y_test = y_test.astype(np.int32)
        
    if subtract_mean & load_train:
        X_train = X_train - mean_image
        X_val = X_val - mean_image
        if load_test:
            X_test = X_test - mean_image
    
    return {
      'X_train': X_train,
      'y_train': y_train,
      'X_val': X_val,
      'y_val': y_val,
      'X_test': X_test,
      'y_test': y_test,
      'mean_image': mean_image,
    }


def load_mini_fmow(params, subtract_mean=True, batch_size=1000):
    """
    Load a mini batch of Functional Map of World from the Training dataset. 
    Training: 80% 
    Validation: 10% 
    Test: 10%
    
    Inputs:
    - params: dict containing key parameters of the project
    - subtract_mean: boolean indicating whether to subtract the mean training image
    - batch_size: total number images to load
    
    Returns: A dictionary with the following entries
    - X_train: (batch_size*0.8, 3, 200, 200) array of training images
    - y_train: (batch_size*0.8, ) array of training labels
    - X_val: (batch_size*0.1, 3, 200, 200) array of validation images
    - y_val: (batch_size*0.1, ) array of validation labels
    - X_test: (batch_size*0.1, 3, 200, 200) array of test images
    - y_test: (batch_size*0.1, ) array of test labels; if not available, then 
      y_test will be None
    """
    # Trick to make sure enough data to satisfy batch_size
    load_size = int(batch_size*1.3)
    
    # Load training data, validation data and test data at once
    X_load = np.array([], dtype=np.float32).reshape(0, 1, 224, 224)
    y_load = np.array([], dtype=np.int32).reshape(0, )
    
    batch_dict = {}
    res_ratio = 0.5
    for category in params['fmow_class_names_mini']:
        if category in ['multi-unit_residential', 'single-unit_residential']:
            batch_dict[category] = int(load_size*res_ratio/2)
        else:
            batch_dict[category] = int(load_size*(1-res_ratio)/(len(params['fmow_class_names_mini'])-2))
    
#     print("Batch Counts: ")
#     for k, v in batch_dict.items():
#         print(k, v)
    
    for category in tqdm(params['fmow_class_names_mini']):
        print("Extracting images from %s..." % category)
        path = os.path.join(params['dataset'], 'train', category)
        X_load, y_load = _process_dir(params, path, category, X_load, y_load, batch_dict)
    
    X_load = X_load.astype(np.float32)
    y_load = y_load.astype(np.int32)
    
    indices = np.random.choice(X_load.shape[0], batch_size, replace=False)
    
    X_train = X_load[indices[:int(batch_size*0.8)], :, :, :]
    y_train = y_load[indices[:int(batch_size*0.8)], ]
    X_val = X_load[indices[int(batch_size*0.8):int(batch_size*0.9)], :, :, :]
    y_val = y_load[indices[int(batch_size*0.8):int(batch_size*0.9)], ]
    X_test = X_load[indices[int(batch_size*0.9):batch_size], :, :, :]
    y_test = y_load[indices[int(batch_size*0.9):batch_size], ]
    
    mean_image = np.mean(X_train, axis=0)
    if subtract_mean:
        X_train = X_train - mean_image
        X_val = X_val - mean_image
        X_test = X_test - mean_image
    
    return {
      'X_train': X_train,
      'y_train': y_train,
      'X_val': X_val,
      'y_val': y_val,
      'X_test': X_test,
      'y_test': y_test,
      'mean_image': mean_image,
    }

def load_test(params, mean_image):
    X_test = np.array([], dtype=np.float32).reshape(0, 1, 224, 224)
    y_test = np.array([], dtype=np.int32).reshape(0, )
    
    print("Extracting Test Images...")
    
    path = os.path.join(params['dataset'], 'test')
    counts = {}
    counts[""] = sys.maxsize
    X_test, y_test = _process_dir(params, path, "", X_test, y_test, counts)  
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.int32)
    if mean_image is not None:
        X_test = X_test - mean_image
        
    return {
      'X_test': X_test,
      'y_test': y_test,
    }

def _process_dir(params, path, category, X, y, counts):
    '''
    Load images and labels from 'category' directory
    
    Inputs:
    - paras: dict of key parameters of project
    - category: class name to be processed
    - X: (M, 3, 200, 200) of images
    - y: (M, ) of labels
    - counts: dict specifying maximum number of images for each category
    
    Returns:
    - X: (M+k, 3, 200, 200) of images
    - y: (M+k, ) of labels
    '''
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('_rgb.json'):
                metadata = os.path.join(root, file)
                image = file[:-5] + '.jpg'
                image = os.path.join(root, image)
                if not os.path.isfile(image):
                    continue
                # Populate the X_train and y_train with image and metadata file
                X, y = _process_image(params, image, metadata, X, y)
                counts[category] = counts[category] - 1     
                if counts[category] <= 0:
                    return (X, y)
    return (X, y)            

def _process_image(params, image_file, metadata_file, X, y):
    """
    Private function to populate X_train and y_train based on image and metadata.
    Need to crop the image by bounding boxes given in metadata. 
    
    Inputs:
    - image: JPG image file to be transformed into (200, 200, 3) array of image
    - metadata: JSON file that specifies detailed information regarding image
    - X: (N, 3, 200, 200) array of images
    - y: (N, ) array of image labels. If category is not given in metadata, pass
    """
       
    # Try acquiring the image and metadata json file
    try:
        image = cv2.imread(image_file, 0).astype(np.float32)
        m = open(metadata_file)
        metadata = json.load(m)
        m.close()
    except Exception as e:
        print("Exception: %s" % e)
        return (X, y)
    
    # Turn metadata['bounding_boxes'] into a list if it is not already
    if not isinstance(metadata['bounding_boxes'], list):
        metadata['bounding_boxes'] = [metadata['bounding_boxes']]
    for bb in metadata['bounding_boxes']:
        # box: [x, y, width, height]
        box = bb['box']
        # skip tiny box
        if box[2] <= 2 or box[3] <= 2:
            continue

        # train with context around box
        
#         contextMultWidth = 0.15
#         contextMultHeight = 0.15
        
#         wRatio = float(box[2]) / image.shape[1]
#         hRatio = float(box[3]) / image.shape[0]
        
#         if wRatio < 0.5 and wRatio >= 0.4:
#             contextMultWidth = 0.2
#         if wRatio < 0.4 and wRatio >= 0.3:
#             contextMultWidth = 0.3
#         if wRatio < 0.3 and wRatio >= 0.2:
#             contextMultWidth = 0.5
#         if wRatio < 0.2 and wRatio >= 0.1:
#             contextMultWidth = 1
#         if wRatio < 0.1:
#             contextMultWidth = 2
            
#         if hRatio < 0.5 and hRatio >= 0.4:
#             contextMultHeight = 0.2
#         if hRatio < 0.4 and hRatio >= 0.3:
#             contextMultHeight = 0.3
#         if hRatio < 0.3 and hRatio >= 0.2:
#             contextMultHeight = 0.5
#         if hRatio < 0.2 and hRatio >= 0.1:
#             contextMultHeight = 1
#         if hRatio < 0.1:
#             contextMultHeight = 2
        
#         widthBuffer = int((box[2] * contextMultWidth) / 2.0)
#         heightBuffer = int((box[3] * contextMultHeight) / 2.0)

        buffer = 16
        r1 = box[1] - buffer
        r2 = box[1] + box[3] + buffer
        c1 = box[0] - buffer
        c2 = box[0] + box[2] + buffer

        if r1 < 0:
            r1 = 0
        if r2 > image.shape[0]:
            r2 = image.shape[0]
        if c1 < 0:
            c1 = 0
        if c2 > image.shape[1]:
            c2 = image.shape[1]

        if r2-r1 <= 5 or c2-c1 <= 5:
            continue

        subimg = image[r1:r2, c1:c2]
        subimg = cv2.resize(subimg, (224, 224)).astype(np.uint8)
        
        if 'category' not in bb:
            rbc_category = -1
        else:
            fmow_category = params['fmow_class_names'].index(bb['category'])
            if fmow_category in [30, 48]:
                rbc_category = 0
            else:
                rbc_category = 1
       
        subimg = np.expand_dims(subimg, axis=0)
        subimg = np.expand_dims(subimg, axis=0)
        X = np.concatenate((X, subimg), axis=0)
        y = np.concatenate((y, np.array([rbc_category])), axis=0)
    return (X, y)
    

    
    

# Below are functions defined for Stanford CS231n. Take them as needed.

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
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
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
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

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

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
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
