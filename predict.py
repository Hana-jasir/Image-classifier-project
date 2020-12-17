
from train import *
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from collections import OrderedDict
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import PIL
import json
import argparse
import seaborn as sns


def arg_parser():
    parser = argparse.ArgumentParser(description="Training Image Classifier Project")

    parser.add_argument('--save_dir', default="./checkpoint.pth", type = str)

    parser.add_argument('--gpu', action='store_true', default="gpu")

    parser.add_argument('--image', type= str, default = 'flowers/test/20/image_04910.jpg')

    parser.add_argument('--cat_file', type= str, default = 'cat_to_name.json')

    parser.add_argument('--top_k', type=int, default =5)

    args = parser.parse_args()

    return args

def load_checkpoint(filepath):

    checkpoint = torch.load(filepath)

    arch = checkpoint['architechture']

    model = models.__dict__[arch](pretrained=True)

    # stop gradients
    for param in model.parameters(): param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    epochs = checkpoint['epochs']
    

    
    return model



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    img = PIL.Image.open(image)
    
    width, height = img.size
    ratio_h = float(height) / float(width)
    ratio_w = float(width) / float(height)

    if width > height:
        new_height = ratio_w * 256
        img = img.resize((256, int(new_height)), Image.ANTIALIAS)
        
    else:
        new_width = ratio_h * 256
        img = img.resize((int(new_width), 256), Image.ANTIALIAS)
    
    width, height = img.size

    left = (width/2 - 224/2)
    top = (height/2 - 224/2)
    right = (width/2 + 224/2)
    bottom = (height/2 + 224/2)
    
    img = img.crop((left, top, right, bottom))

    numpy_img = np.array(img)/255 

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    numpy_img = (numpy_img-mean)/std
 
    numpy_img = numpy_img.transpose(2, 0, 1)
    
    return numpy_img


def predict(image_path, model, cat_file, top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    
    image_path: string. Path to image, directly to image and not to folder.
    model: pytorch neural network.
    top_k: integer. The top K classes to be calculated
    
    returns top_probabilities(k), top_labels
    '''
    with open(cat_file, 'r') as f:
        cat_to_name = json.load(f)

    model.to("cpu")
    model.eval();

    # Convert image from numpy to torch
    image = process_image(image_path)
   
    torch_image = torch.from_numpy(image).type(torch.FloatTensor).to('cpu')
   
    torch_image = torch_image.unsqueeze (dim = 0)
   

    with torch.no_grad ():
        logps = model.forward(torch_image)
  
    outputs  = torch.exp(logps)
 
    top_probs, top_indices  = outputs.topk(top_k)
 
    
    top_probs = top_probs.numpy().tolist()[0]
 
    top_indices = top_indices.numpy().tolist()[0]
    
    # Convert to classes
    idx_to_class = {value: key for key, value in
                        model.class_to_idx.items()}

    top_labels = [idx_to_class[idx] for idx in top_indices]
    top_flowers = [cat_to_name[label] for label in top_labels]

    return top_probs, top_labels, top_flowers


def print_probs(top_probs, flowers):

    print("Top five classes are:\n")
    n = 0
    for x, y in zip(top_probs, flowers):
        n +=1
        print("{}. {:30} , Probability: {:.3f}%".format(n, y, x*100))


def main():

    args = arg_parser()

    device = check_gpu(args.gpu)

    model = load_checkpoint(args.save_dir)

    top_probs, top_classes, top_flowers = predict(args.image, model,
                args.cat_file, args.top_k)

    print_probs(top_probs, top_flowers)



if __name__ == "__main__":
    main()
