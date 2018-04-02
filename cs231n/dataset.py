from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class FMOWDataset(Dataset):
    """Functional Map of World Dataset"""

    def __init__(self, params, transform=None):
        """
        Args:
            params (dict): Dict containing key parameters of the project
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.params = params
        self.transform = transform
        self.map = {}
        self.curr_index = 0
        def populate_map(category):
            path = os.path.join(params['dataset'], 'train', category)
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('_rgb.jpg'):
                        image = os.path.join(root, file)
                        if not os.path.isfile(image):
                            continue
                        self.map[self.curr_index] = image
                        self.curr_index += 1
            
        for category in params['fmow_class_names_mini']:
            populate_map(params)     

    def __len__(self):
        return len(self.map)

    def __getitem__(self, idx):
        image_p = self.map[idx]
        metadata_p = image[:-3] + '.json'
        try:
            image = cv2.imread(image_p, 0).astype(np.float32)
            m = open(metadata_p)
            metadata = json.load(m)
            m.close()
        except Exception as e:
            print("Exception: %s" % e)
        if isinstance(metadata['bounding_boxes'], list):
            metadata['bounding_boxes'] = metadata['bounding_boxes'][0]
        # Train only on first bounding box
        bb = metadata['bounding_boxes']
        
        box = bb['box']
        buffer = 16
        r1 = box[1] - buffer
        r2 = box[1] + box[3] + buffer
        c1 = box[0] - buffer
        c2 = box[0] + box[2] + buffer
        r1 = max(r1, 0)
        r2 = min(r2, image.shape[0])
        c1 = max(c1, 0)
        c2 = min(c2, image.shape[1])
        image = image[r1:r2, c1:c2]
        if self.transform:
            sample = self.transform(subimg)
        
        fmow_cat = params['fmow_class_names'].index(bb['category'])
        if fmow_category in [30, 48]:
            label = 0
        else:
            label = 1     
            
        return image, label
    
class ToTensor(object):
    """Convert ndarrays to Tensors."""

    def __call__(self, image, label):
        # Move color channel
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image), torch.from_numpy(label)
    
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple): Desired output size. Output is
            matched to output_size
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (tuple))
        self.output_size = output_size

    def __call__(self, image, label):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        image = cv2.resize(subimg, (new_w, new_h))
        return image, label
    
    
    

