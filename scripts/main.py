# Import necessary libraries
import yaml  # For loading configuration files
import time  # For time-related operations
import os  # For operating system related operations
import sys  # For system-specific parameters and functions
from PIL import Image  # For image processing
import re  # For regular expressions

import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis
from matplotlib import pyplot as plt  # For plotting

import torch  # For PyTorch operations
import torch.nn as nn  # For neural network modules in PyTorch
from torch.utils.data import TensorDataset, DataLoader, Dataset  # For dataset and data loading utilities in PyTorch
from torchvision import transforms  # For image transformations

from sklearn.model_selection import train_test_split  # For splitting data into train and test sets

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# Append necessary directories to system path
sys.path.append(os.path.join(parent_dir, 'data'))  # Append data directory to system path
sys.path.append(os.path.join(parent_dir, 'models'))  # Append models directory to system path
sys.path.append(os.path.join(parent_dir, 'config'))  # Append config directory to system path
sys.path.append(os.path.join(parent_dir, 'utils'))  # Append utils directory to system path

# Import custom modules
from data_loader import load_data_files  # Custom data loader
from model import ID, MAB, IDMAB, IDMAG, DFES, SFES, PixelShuffle, ReconstructionBlock, FeedbackBlock, IDMAN, params  # Custom neural network modules
from train import train  # Custom training function
from evaluation import evaluation  # Custom evaluation function
from Instance_visualization import Instance_visualization  # Custom layer visualization function

# Load configuration settings from YAML file
with open(os.path.join(parent_dir, 'config', 'config.yml'), 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

# Define directories for training and testing data
directory = os.path.join(parent_dir, 'data', 'dataset')

print('Data is loading ... !')

# Load training and testing data based on configuration settings
size_train = config['size_train']
size_val = config['size_val']
size_test = config['size_test']
test_image_size = config['test_image_size']
train_crop_size = config['train_crop_size']

train_data_loader, valid_data_loader, test_data_loader = load_data_files(
    directory, train_size=size_train, val_size=size_val, test_size=size_test, image_size=test_image_size, patch_crop=train_crop_size)

# Print the shape of a batch of test data
#x, y = next(iter(test_data_loader))
#print(x.shape, y.shape)

print('Data is loaded !')
print()

# Check if pretraining is enabled
Pretrain = config['Pretrain']

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if Pretrain == False:
    # Load model parameters from configuration
    num_input = config['num_input']
    num_features = config['num_features']
    B_num = config['B_num']
    G_num = config['G_num']
    mba_r = config['mba_r']
    mba_n = config['mba_n']
    mba_kn = config['mba_kn']
    
    # Initialize model with the parameters
    parameters = params(num_input=num_input, num_features=num_features, B=B_num, G=G_num, mba_r=mba_r, mba_n=mba_n, mba_kn=mba_kn)
    model = IDMAN(parameters)
    model.to(device)
    #print(mba_kn)
    print()

    # Load training settings from configuration
    lr = config['lr']
    step_size = config['step_size']
    gamma = config['gamma']
    epoch_number = config['epoch_number']

    # Set loss function based on configuration
    if config['loss'] == 'L1':
        lossfun = nn.L1Loss()
    if config['loss'] == 'MSE':
        lossfun = nn.MSELoss()
    
    # Initialize optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Train the model
    train(model, device, train_data_loader, valid_data_loader, lossfun, optimizer, scheduler, epoch_number=epoch_number, 
            save_model=False, model_name='Unknown', visualization=True)

if Pretrain == True:
    # Load pretrained model
    if config['loss'] == 'L1':
        PATH = os.path.join(parent_dir, 'models', 'saved_models', 'base_model_Final_project.pth')
    if config['loss'] == 'MSE':
        PATH = os.path.join(parent_dir, 'models', 'saved_models', 'base(withMSE)_model_Final_prject.pth')
    model = torch.load(PATH, map_location=device) 

# Evaluate the model on test data
print('Model Performance on test set')
evaluation(model, device, test_data_loader)

# Visualize instances
Instance_visualization(model, device, test_data_loader, image_size=test_image_size)

# Display any generated plots
plt.show()
