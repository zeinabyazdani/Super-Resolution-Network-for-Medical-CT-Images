import numpy as np
import sys
import time
from matplotlib import pyplot as plt
import torch

def train(model, device, train_loader, valid_loader, lossfun, optimizer, scheduler, epoch_number=15, 
          save_model=False, model_name='Unknown', visualization=True):
  
  loss_train_epoch = []  # List to store training losses for each epoch
  loss_valid_epoch = []  # List to store validation losses for each epoch


  training_time = 0      # Variable to track total training time

  model.to(device)       # Move the model to the specified device (GPU or CPU)

  for epoch in range(epoch_number):
    loss_train_batch = 0   # Variable to accumulate training loss for each batch
    acc_train_batch = 0    # Variable to accumulate training accuracy for each batch
    num_train_batches = 0 # Total number of batches in training set

    # Training loop
    model.train()  # Set the model to train mode
    for X, Y in train_loader:
      X = X.to(device)  # Move input data to the specified device
      Y = Y.to(device)  # Move target labels to the specified device

      start_time = time.time()  # Record start time for batch processing
      yHat = model(X)        # Forward pass
      loss = lossfun(yHat, Y)  # Calculate loss
      
      optimizer.zero_grad()  # Clear gradients of optimizer
      loss.backward()         # Backpropagation
      optimizer.step()        # Update weights

      loss_train_batch += loss.item()  # Accumulate training loss
      num_train_batches += 1  # Increment the number of batches processed
      stop_time = time.time()   # Record stop time for batch processing
      training_time += (stop_time - start_time)  # Accumulate batch processing time
      moving_batch_loss = loss_train_batch / num_train_batches
      sys.stdout.write('\r' + f'epoch = {epoch + 1} , batch = {num_train_batches}, loss Train = {moving_batch_loss:.4f}')

    loss_train_epoch.append(moving_batch_loss)  # Calculate average training loss for epoch
    scheduler.step()  # Adjust learning rate scheduler
    # Validation loop
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
      loss_valid_batch = 0   # Variable to accumulate validation loss for each batch
      acc_valid_batch = 0    # Variable to accumulate validation accuracy for each batch
      num_valid_batches = 0  # Total number of batches in validation set

      for X, Y in valid_loader:
        X = X.to(device)   # Move input data to the specified device
        Y = Y.to(device)   # Move target labels to the specified device

        yHat = model(X)    # Forward pass
        loss = lossfun(yHat, Y)  # Calculate loss

        loss_valid_batch += loss.item()  # Accumulate validation loss
        num_valid_batches += 1  # Increment the number of batches processed
        moving_batch_loss = loss_valid_batch / num_valid_batches
    loss_valid_epoch.append(moving_batch_loss)  # Calculate average validation loss for epoch

    # Print progress
    print('\r' + f'epoch = {epoch + 1}, loss Train = {loss_train_epoch[-1]:.4f}, loss Valid = {loss_valid_epoch[-1]:.4f}')

  if save_model:
    PATH = model_name + '.pth'
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    torch.save(model, PATH) # Save model parameters to file
    
  print()   # Print newline for better readability
  print(f'Training took {training_time:.4f} seconds')  # Print total training time

  if visualization:
    # Plot training and validation loss
    plt.figure()
    plt.plot(np.arange(epoch_number), loss_train_epoch, '-*', color='blue', label='Train')
    plt.plot(np.arange(epoch_number), loss_valid_epoch, '-o', color='orange', label='Validation')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title(f'Final Validation Loss Value = {loss_valid_epoch[-1]:.4f}')
