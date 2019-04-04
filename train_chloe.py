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


# Define command line arguments
parser = argparse.ArgumentParser(description='Train a model to classify types of flowers')
parser.add_argument('--data_dir', type=str,default='flowers', help='Filepath')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
parser.add_argument('--epochs', type=int, default=4,help='# for epochs')
parser.add_argument('--arch', type=str, default='VGG', help='Model architecture')
parser.add_argument('--learning_rate', default=0.0003,type=float, help='Learning rate')
parser.add_argument('--hidden_units', type=int,default=1000,help='Number of hidden units')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pth',help='Save trained model checkpoint to file')


# Store user options in the "args" variable
args = parser.parse_args()   

data_dir = args.data_dir #data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder 
# TODO: Load the datasets with ImageFolder
train_set = datasets.ImageFolder(train_dir, transform=train_transforms)
validation_set = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_set = datasets.ImageFolder(test_dir, transform=valid_transforms)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=32,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32,shuffle=True)

# The training script allows users to choose from at least two different architectures available from torchvision.models
if args.arch == 'VGG':
    model = models.vgg19(pretrained=True)
    num_feat = model.classifier[0].in_features #input #
else:
    model = models.densenet161(pretrained=True)
    num_feat = model.classifier.in_features
    
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# Build classifier
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(num_feat, 4096)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(4096, args.hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.5)),
                          ('fc3', nn.Linear(args.hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
 
    
model.classifier = classifier

# Set negative log loss as the criteria
criterion = nn.NLLLoss()
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate) # Need to give the option to change learn rate

# Train the network
def train(model, trainloader, testloader, criterion, optimizer, epochs=5, print_every=40):

    steps = 0
    running_loss = 0

    # change to cuda
    model.to('cuda')

    for epoch in range(epochs):
        running_loss = 0
        for ii, (images, labels) in enumerate(trainloader):
            steps += 1

            images, labels = images.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                v_lost = 0
                v_accuracy=0
                for ii, (images2,labels2) in enumerate(testloader):
                    optimizer.zero_grad()
                    images2, labels2 = images2.to('cuda') , labels2.to('cuda')
                    model.to('cuda')
                    with torch.no_grad():    
                        outputs = model.forward(images2)
                        v_lost = criterion(outputs,labels2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        v_accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
                v_lost = v_lost / len(testloader)
                v_accuracy = v_accuracy /len(testloader)
            
                print("Epoch: {}/{}... ".format(epoch+1, epochs),
                  "Loss: {:.4f}".format(running_loss/print_every),
                  "Validation Lost {:.4f}".format(v_lost),
                   "Accuracy: {:.4f}".format(v_accuracy))
                running_loss = 0

#training
#Prints out training loss, validation loss, and validation accuracy as the network trains
train(model, train_loader, validation_loader, criterion, optimizer, epochs=args.epochs, print_every=40)

                
# testing the model
def test_test(model, testloader, criterion):
    accuracy = 0
    total = 0
    model.eval() 
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    print('Accuracy on the test: %d %%' % (100 * accuracy / total))
    
# TODO: Do validation on the test set
test_test(model, test_loader, criterion)


# TODO: Save the checkpoint
model.class_to_idx =  train_set.class_to_idx
model.cpu
torch.save({'input_size': num_feat,
            'output_size': 102,
            'state_dict':model.state_dict(),
            'class_to_idx':model.class_to_idx},
            'checkpoint.pth')

