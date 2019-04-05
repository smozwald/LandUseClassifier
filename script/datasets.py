from skimage.io import imread
from PIL import Image ##For the output mask

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as tf

class EOLearnDataset(Dataset):
  """This will process the images into beautiful tensors for learning.
  Need to work the classes a bit better methinks.
  Was used with the 6-band pre-transformed images of EO slovenia."""
  def __init__(self,images, target_dir, classes, class_ints = None, invalid_classes = None):
    """Images = List of images to add, in path format.
    target_dir = Target directory, where y images can also be appended.
    Classes are tied to integers in mask image, may require fine tuning for working classes and which are included as valid/invalid."""
    self.images = images
    self.target_dir = target_dir
    
    self.valid_classes = classes ##Need to make this also work with None
    if not self.valid_classes:
      pass
    self.class_ints = class_ints
    if not self.class_ints:
      self.class_ints = list(range(len(self.valid_classes)))
    self.invalid_class_ints = invalid_classes
    ##Remove invalid_classes from valid ones
      
    self.class_map = dict(zip(self.valid_classes,self.class_ints))
    
  def transform(self, image, targ):
    """Make some random flips and stuff."""
    """ if random.random() > 0.5:
      image = tf.hflip(image)
      targ = tf.hflip(targ)
    
    if random.random() > 0.5:
      image = tf.vflip(image)
      targ = tf.vflip(targ)"""
    
    image = tf.to_tensor(image)
    #image = tf.normalize(image, 0, 1) ##Normalize input
    targ = tf.to_tensor(targ).type(torch.long)
    return image, targ
  
  def get_y(self, image_file, target_dir):
    """Returns target image for image."""
    return os.path.join(target_dir, image_file.split("/")[-1])
    
  def encode_segmap(self, targ):
    """Segementing target array to classes.
    I think this is not actually needed since the images are encoded and i am a big dum dum.
    Maybe for other images which aren't encoded like this."""
    mask = np.zeros(targ.shape, dtype = int)
    for vclass in self.class_ints:
      mask[targ == vclass] = self.class_map[vclass]
    for iclass in self.invald_class_ints:
      mask[targ == iclass] = self.ignore_index
    return mask
    
  def __len__(self):
    return len(self.images)
  
  def __getitem__(self, idx):
    img = imread(self.images[idx])
    targ = imread(self.get_y(self.images[idx], self.target_dir))
    #targ = self.encode_segmap(targ)
    targ = Image.fromarray(np.int8(targ))
    x, y = self.transform(img, targ)
    sample = {"x": x, "y":y}
    return x,y