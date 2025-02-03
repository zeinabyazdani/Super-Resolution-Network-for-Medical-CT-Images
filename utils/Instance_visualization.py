from matplotlib import pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image

def intensity_windowing(tensor, window_min, window_max, output_min=0, output_max=1):

    # Applies intensity windowing to a tensor


    # Clip the tensor values to the window range
    clipped_tensor = torch.clamp(tensor, window_min, window_max)
    
    # Normalize the clipped tensor to the range [0, 1]
    normalized_tensor = (clipped_tensor - window_min) / (window_max - window_min)
    
    # Scale the normalized tensor to the output range [output_min, output_max]
    scaled_tensor = normalized_tensor * (output_max - output_min) + output_min
    
    return scaled_tensor

def Instance_visualization(model,device,data,image_size = 512, number_of_instance = 1, window_min_par = 0.4, window_max_par = 0.7):

  # Visualizes the output of a model along with the bicubic reconstruction and differences.

  sample = 0
  for x , y in data:
      if sample < number_of_instance:
          sample +=1
          # Generate model output and detach it from computation graph
          output = model(x.to(device)).cpu().detach()
          inpt = x
          preds = output
          target = y

          # Resize the input image using bicubic interpolation
          Resizer = transforms.Resize((image_size,image_size) , interpolation=Image.BICUBIC, antialias=True)
          BICUBIC_image = Resizer(inpt)

          # Compute the absolute differences
          BICUBIC_diff = abs(BICUBIC_image - target).squeeze().numpy()
          SuperResolution_diff = abs(preds - target).squeeze().numpy()

          # Apply intensity windowing to the images
          BICUBIC_corrected = intensity_windowing(BICUBIC_image, window_min = window_min_par, window_max = window_max_par)
          preds_corrected = intensity_windowing(preds, window_min = window_min_par, window_max = window_max_par)


          # Apply intensity windowing to the images
          plt.figure(figsize = (10,10))
          plt.subplot(2,2,1)
          plt.imshow(BICUBIC_corrected.squeeze(),cmap = 'gray')
          plt.axis('off')
          plt.title('BICUBIC Reconstruction')

          plt.subplot(2,2,2)
          plt.imshow(preds_corrected.squeeze(),cmap = 'gray')
          plt.axis('off')
          plt.title('SuperResolution Reconstruction')

          plt.subplot(2,2,3)
          plt.imshow(BICUBIC_diff,cmap = 'gray')
          plt.axis('off')
          plt.title('BICUBIC difference')

          plt.subplot(2,2,4)
          plt.imshow(SuperResolution_diff,cmap = 'gray')
          plt.axis('off')
          plt.title('SuperResolution difference')
          plt.show()
