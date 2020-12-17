
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from collections import OrderedDict
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import PIL
import argparse
import seaborn as sns
import json

# ## Load the data
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


train_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                      transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(), 
                                     transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256), 
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
valid_transforms = transforms.Compose([transforms.Resize(256), 
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=64)
testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64)



with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)



# TODO: Build and train your network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg19(pretrained=True)

# stop computing gradients
for param in model.parameters():
    param.requires_grad = False

# create new classifier
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 1024)),
                          ('drop', nn.Dropout(p=0.05)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(1024, 512)),
    ('relu2', nn.ReLU()),
    ('drop2', nn.Dropout(p=0.05)),
    ('fc4', nn.Linear(512, 102)),
    
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

# attach the model to a classifier   
model.classifier = classifier

#define my loss
criterion = nn.NLLLoss()

#define my learning rate
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

model.to(device)

steps = 0
running_loss = 0
print_every = 20 # how many steps will print out before valid loss
epochs = 7
test_loss = 0
accuracy = 0
#model.eval()???

for epoch in range(epochs):
    for images, labels in trainloader:#training loop
        steps += 1 #every time we're going to go through one of the batches, steps will be increased
        images, labels = images.to(device), labels.to(device)#move images and labels to the gpu
        
        optimizer.zero_grad()
        
        logps = model.forward(images)
        loss = criterion(logps, labels)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
     
        if steps % print_every == 0:# valid loop
            valid_loss = 0
            accuracy = 0
            model.eval()
            
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # accuracy calc
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()#update accuracy
                    
            print("Epoch: {}/{} | ".format(epoch+1, epochs),
                      "Training Loss: {:.3f} | ".format(running_loss/print_every),
                      "Validation Loss: {:.3f} | ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
            running_loss = 0
            model.train()


# Do validation on the test set
test_loss = 0
accuracy = 0


with torch.no_grad():
    model.eval()
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)
        test_loss += batch_loss.item()
        
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
    print("Test loss: {:.4f}".format(test_loss/len(testloader)),"test accuracy: {:.4f}".format(accuracy/len(testloader))  ) 





# Save the checkpoint 
model.class_to_idx = train_datasets.class_to_idx
            
            # Create checkpoint dictionary
checkpoint = {
              'classifier': model.classifier,
              'class_to_idx': model.class_to_idx,
              'state_dict': model.state_dict(),
              'epochs': epochs, 
              'optimizer_state_dict': optimizer.state_dict()}
    
            
torch.save(checkpoint, 'checkpoint.pth')

# a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    model = models.vgg19(pretrained=True)
    
    # stop gradients
    for param in model.parameters(): param.requires_grad = False
    
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    epochs = checkpoint['epochs']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    
    return model

model = load_checkpoint('checkpoint.pth')


def process_image(image):

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



def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax



def predict(image_path, model, top_k=5):

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


# TODO: Display an image along with the top 5 classes

image_path = "flowers/test/20/image_04910.jpg"

probs, lables, flowers = predict(image_path, model) 

plt.figure(figsize = (5,10))
ax = plt.subplot(2,1,1)
plt.subplot(2,1,2)

title = cat_to_name[image_path.split('/')[-2]]

np_img = process_image(image_path)
imshow(np_img, ax, title = title);

sns.barplot(x=probs, y=flowers, color=sns.color_palette()[0]);
plt.show()
 