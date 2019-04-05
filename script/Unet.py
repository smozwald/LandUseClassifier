"""Created by Samuel Oswald in March 2019.
This is an implementation of Unet for image segmentation,
based on some other repositories (see LandUseClassifier-Slovenia notebook).
Also includes some functions for classifying single images for output."""

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as tf
import torch.nn as nn
import torch.nn.functional as F

def predict_single_image(model, data):
  """Should contain a tensor and correct output as input, will return prediction and ground truth numpy arrays.
  Input will be converted to a dataloader and then model applied for prediction."""
  loader = DataLoader([data])
  for X,y in loader:
    X = X.to('cpu', dtype = torch.float)
    pred = model.to('cpu')(X)
    max_pred = torch.argmax(pred, dim =1).cpu().numpy().squeeze(0)
    y = y.cpu().numpy().squeeze(0)
  return max_pred, y.squeeze(0)

class UNetDownBlock(nn.Module):
  """As the UNet contracts, it does so with a bunch of these blocks effectively."""
  def __init__(self, in_size, out_size, padding, batch_norm):
    super(UNetDownBlock, self).__init__()
    block = []
    
    block.append(nn.Conv2d(in_size, out_size, kernel_size = 3,
                          padding = int(padding)))
    block.append(nn.ReLU())
    if batch_norm:
      block.append(nn.BatchNorm2d(out_size))
      
    block.append(nn.Conv2d(out_size, out_size, kernel_size = 3,
                          padding = int(padding)))
    block.append(nn.ReLU())
    if batch_norm:
      block.append(nn.BatchNorm2d(out_size))
      
    self.block = nn.Sequential(*block)
    
  def forward(self, x):
    out = self.block(x)
    return out

class UNetUpBlock(nn.Module):
  """As the UNet expands, it fills out to get back towards the original size."""
  def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
    super(UNetUpBlock, self).__init__()
    if up_mode == 'upconv':
      self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
    
    elif up_mode == 'upsample':
      self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))
    
    self.conv_block = UNetDownBlock(in_size, out_size, padding, batch_norm)
    
  def center_crop(self, layer, target_size):
    _, _, layer_height, layer_width = layer.size()
    diff_y = (layer_height - target_size[0]) // 2
    diff_x = (layer_width - target_size[1]) // 2
    return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

  def forward(self, x, bridge):
    up = self.up(x)
    crop1 = self.center_crop(bridge, up.shape[2:])
    out = torch.cat([up, crop1],1)
    out = self.conv_block(out)

    return out

class UNet(nn.Module):
  def __init__(self, in_channels=1, n_classes = 2, depth = 5, factor = 6, padding = True,
              batch_norm = False, up_mode = 'upconv'):
    """Implementation of UNet. Arguments as follows:
    in-channels: number of input channels, would be 1 for a single image, however in our case 6, should be read from tensor.
    n_classes (int). The output targets, in our initial example case is 10 for 10 land cover classes.
    depth: how deep the network should go.
    factor: The first layer filters is 2**factor, as is the same for every other layer.
    padding: To get output image of same dimensions (pretty handy for land cover classification,
    since you'll probably want to use it for GPS and such, although not necessary.)
    batch_norm: To use BatchNorm after activation functions."""
    
    super(UNet, self).__init__()
    assert up_mode in ('upconv','upsample')
    self.padding = padding
    self.depth = depth
    prev_channels = in_channels
    ##Create our list of modules for downwards
    self.down_path = nn.ModuleList()
    for i in range(depth):
      self.down_path.append(UNetDownBlock(prev_channels, 2**(factor+i),
                                         padding, batch_norm))
      prev_channels = 2**(factor+i)
      
    ##Create our upsampling modules, note that we will ned to reference our downsample models.
    self.up_path = nn.ModuleList()
    for i in reversed(range(depth-1)):
      self.up_path.append(UNetUpBlock(prev_channels, 2**(factor + i), up_mode,
                                     padding, batch_norm))
      prev_channels = 2** (factor+i)
    
    ##Get output classes
    self.last = nn.Conv2d(prev_channels, n_classes, kernel_size = 1)
    
  def forward(self, x):
    blocks = []

    for i, down in enumerate(self.down_path):
      x = down(x)
      if i != len(self.down_path) -1: ##So when it's not the final block, since that time we just upsample without another pool.
        blocks.append(x)
        x = F.max_pool2d(x, 2)

    for i, up in enumerate(self.up_path):
      x = up(x, blocks[-i-1]) ##Get the skip connection from before.

    return self.last(x)