import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset , DataLoader



class CustomDataset(Dataset):
    #Custom dataset for handling image data with optional augmentation and cropping. 

    def __init__(self, images, image_size ,crop = True, Augmentation = True , scale = 2):
        self.images = images
        self.image_size = image_size
        self.crop = crop
        self.Augmentation = Augmentation
        self.rotation_angles = [90, 180, 270]
        self.scale = scale
        self.LR_size = int(image_size/scale)

        # Define augmentation transformations
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
        ])
        
        # Define cropping transformation   
        self.crop = transforms.Compose([
            transforms.RandomCrop(self.image_size),
        ]) 

        # Define downsize transformation            
        self.downsize_transform = transforms.Compose([
            transforms.Resize((self.LR_size) , interpolation=Image.BICUBIC, antialias=True),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load the image
        image_path = self.images[idx]
        image = Image.open(image_path)
       
        # Apply rotation and augmentation randomly
        if self.Augmentation:
            angle = np.random.choice(self.rotation_angles)
            image = transforms.functional.rotate(image, int(angle))
            image = self.augmentation(image)
        
        # Convert image to tensor and normalize
        image = transforms.ToTensor()(image)
        image = (image  - torch.min(image)) / (torch.max(image)  - torch.min(image))
  
        # Apply cropping if specified
        if self.crop:
            image = self.crop(image)
        output_image = image
        input_image = self.downsize_transform(image)
        

        return input_image, output_image

def load_data_files(directory,train_size=10000,val_size=500,test_size=1000,image_size = 512, patch_crop=48):
    # Loads and splits data files into training, validation, and test sets.
    """
    Parameters:
    directory (str): Path to the directory containing images.
    train_size (int): Number of training samples. Default is 10000.
    val_size (int): Number of validation samples. Default is 500.
    test_size (int): Number of test samples. Default is 1000.
    image_size (int): Size to which test images will be resized. Default is 512.
    patch_crop (int): Size of cropped patches for training and validation. Default is 48.

    """
    
    file_names = []

    # Calculate total size
    total_size = train_size + val_size + test_size

    # Collect image file paths
    for direc in os.listdir(directory):
        if len(file_names) >= total_size:
            break
        image_path = os.path.join(directory, direc)
        if (image_path.endswith(".png")):
            file_names.append(image_path)

    # Split the data into train, validation, and test sets
    data_train_temp, data_test = train_test_split(file_names, test_size = test_size, random_state=42)
    data_train, data_valid = train_test_split(data_train_temp, test_size = val_size, random_state=42)
    
    # Create datasets and dataloaders
    train_dataset = CustomDataset(data_train , image_size = patch_crop*2 )
    train_dataloader = DataLoader(train_dataset, batch_size= 8, shuffle=True)

    valid_dataset = CustomDataset(data_valid, image_size = patch_crop*2, Augmentation = False)
    valid_dataloader = DataLoader(valid_dataset, batch_size= 8, shuffle=False)

    test_dataset = CustomDataset(data_test, image_size = image_size, Augmentation = False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader  