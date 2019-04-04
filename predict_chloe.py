# Imports here
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms, models
from collections import OrderedDict
import numpy as np
import torch
import time
import json
import glob, os
import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Get input arguments from users
def get_input():
    # Give user the option to choose data directory, save directory, whether to use the GPU or not, the model architecture and so on
    parser = argparse.ArgumentParser(description='Predict flower class using the trained neural net')
    parser.add_argument('--img', type=str, default='flowers/test/10/image_07104.jpg', help='Choose the image to be classified')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth',help='Save trained model checkpoint to file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--topK', type=int, default=5, help='Print out the top K classes with predicted probabilities')
    parser.add_argument('--flower', type=str, default='cat_to_name.json', help='Load the JSON file that maps the class values to category names')
    
    return parser.parse_args()

# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model.to(device)
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model


import math
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    w, h = im.size
    ratio = h / w
    shortenS = 256
    
    if h > w:
        longsize = int(math.floor(shortenS * ratio))
        newsize = (shortenS, longsize)
    else:
        longsize = int(math.floor(shortenS / ratio))
        newsize = (longsize, shortenS)
        
    im = im.resize(newsize)
    
    wn, hn = newsize
    
    left = (wn - 224)/2
    top = (hn - 224)/2
    right = (wn + 224)/2
    bottom = (hn + 224)/2
    
    cropped = im.crop((left, top, right, bottom))
    
    imgplot = plt.imshow(cropped)
    imgplot.axes.get_xaxis().set_visible(False)
    imgplot.axes.get_yaxis().set_visible(False)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = np.array(cropped) / 255
    image = (image - mean) / std
    image = image.transpose((2, 0, 1))
    
    return  torch.from_numpy(image)
    # TODO: Process a PIL image for use in a PyTorch model
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    im = process_image(image_path)
    im.unsqueeze_(0)
    im = im.cuda().float()
    
    model.eval()
    
    with torch.no_grad():
        score = model(im)
        probability, idxs = torch.topk(score, topk)
    
        # convert indices to classes
        idxs = np.array(idxs)            
        idx_to_class = {val:key for key, val in model.class_to_idx.items()}
        classes = [idx_to_class[idx] for idx in idxs[0]]
        
        # map the class name with collected topk classes
        flower = [] 
        for cls in classes: #assign
            flower.append(cat_to_name[str(cls)])
        
        return probability, flower

# TODO: Print the top K classes along with corresponding probabilities
args = get_input()
probability,flower = predict(args.img, args.checkpoint, args.topK)
print('Left: Possible flower   Right: Probability')
for probability, flower in zip(probability, flower):
    print("%20s: %f" % (flower, probability))