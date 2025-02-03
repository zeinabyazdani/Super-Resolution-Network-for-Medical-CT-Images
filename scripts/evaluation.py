import numpy as np
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.image import StructuralSimilarityIndexMeasure

def evaluation(model, device, test_loader):
# Evaluates the performance of a model on a test dataset.

    # Initialize SSIM measure 
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)


    # Initialize lists to store PSNR and SSIM values for each batch
    PSNR_value_total = []
    ssim_value_total = []
    
    # Iterate over the test data
    for x , target in test_loader:
        # Make predictions using the model
        preds = model(x.to(device)).cpu().detach()
        # Compute PSNR between predictions and targets
        PSNR_value = peak_signal_noise_ratio(preds , target)
        # compute SSIM between predictions and targets
        ssim_value = ssim(preds, target)
        # Append the computed values to the lists
        PSNR_value_total.append(PSNR_value)
        ssim_value_total.append(ssim_value)

    # Compute the mean PSNR and SSIM values across all batches
    PSNR_value = np.mean(np.array(PSNR_value_total))
    ssim_value = np.mean(np.array(ssim_value_total))
    # Print the average PSNR and SSIM values
    print(f'PSNR = {PSNR_value:.4f}')
    print(f'SSIM = {ssim_value:.4f}')


