
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
from time import time

def arg_parser():
    parser = argparse.ArgumentParser(description="Training Image Classifier Project")

    parser.add_argument('--data_dir', type= str, default = 'flowers')

    parser.add_argument('--arch', default="vgg16", type = str)

    parser.add_argument('--save_dir', default="./checkpoint.pth", type = str)

    parser.add_argument('--learning_rate', default=0.001, type= int,)

    parser.add_argument('--hidden_1', type=int, default=2024)

    parser.add_argument('--hidden_2', type=int, default=512)

    parser.add_argument('--epochs', type=int, default=7)

    parser.add_argument('--gpu', action='store_true', default="gpu")

    args = parser.parse_args()

    return args

def data_loader(data_dir, train_dir, test_dir, valid_dir):
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

    return trainloader, testloader, validloader, train_datasets


def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cpu":
        print("working on cpu")
    else:
        print("working on gpu")

    return device


def initial_classifier(model, arch, hidden_1, hidden_2, learning_rate, device):

    for param in model.parameters():
        param.requires_grad = False

    input_ = model.classifier[0].in_features


    classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(input_, hidden_1)),
                        ('ReLu1', nn.ReLU()),
                        ('Dropout1', nn.Dropout(p=0.5)),
                        ('fc2', nn.Linear(hidden_1, hidden_2)),
                        ('ReLu1', nn.ReLU()),
                        ('fc3', nn.Linear(hidden_2, 102)),
                        ('output', nn.LogSoftmax(dim=1))
                        ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    model.to(device);

    return model, classifier, criterion, optimizer


def trainer(model, trainloader, validloader, device, epochs, criterion,
                    optimizer, lr, testloader):

    steps = 0
    running_loss = 0
    print_every = 20 # how many steps will print out before valid loss
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

    return model

def save_checkpoint(arch, epochs, save_dir, model, learning_rate, hidden_1, hidden_2, train_dataset, optimizer):
    model.class_to_idx = train_dataset.class_to_idx
            
    checkpoint = {'classifier': model.classifier, 'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'epochs': epochs,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'architechture': arch,}
    
    torch.save(checkpoint, 'checkpoint.pth')

    print("model has been saved")


def main():

    start_time = time()

    args = arg_parser()

    train_dir = 'flowers/train'
    valid_dir = 'flowers/valid'
    test_dir =  'flowers/test'


    train_loader, test_loader, valid_loader, train_dataset = data_loader(
        args.data_dir, train_dir, test_dir, valid_dir)

    device = check_gpu(args.gpu)

    model = models.__dict__[args.arch](pretrained=True)

    model, classifier, criterion, optimizer = initial_classifier(model, args.arch,
        args.hidden_1, args.hidden_2, args.learning_rate, device)


    model = trainer(model, train_loader, valid_loader, device, args.epochs, criterion,
                    optimizer, args.learning_rate, test_loader)

    save_checkpoint(args.arch, args.epochs, args.save_dir, model,
            args.learning_rate, args.hidden_1, args.hidden_2, train_dataset, optimizer)

    end_time = time()
    tot_time = end_time - start_time

    print("\nTotal Elapsed Runtime:", str( int( (tot_time / 3600) ) ) + ":" +
          str( int(  ( (tot_time % 3600) / 60 )  ) ) + ":" +
          str( int(  ( (tot_time % 3600) % 60 ) ) ) )

if __name__ == "__main__":
    main()
