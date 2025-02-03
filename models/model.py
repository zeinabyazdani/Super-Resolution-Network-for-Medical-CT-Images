import torch
import torch.nn as nn
import torch.nn.functional as F



class ID(nn.Module):
    """
    Information Distillation block.
    
    Args:
        in_channels: number of input channels (and output!).
    """
    
    def __init__(self, in_channels):
        super(ID, self).__init__()
        # Split channels
        self.split_channels1 = in_channels // 2
        self.split_channels2 = self.split_channels1 // 2
        self.split_channels3 = self.split_channels2 // 2
        self.split_channels  = self.split_channels1 + self.split_channels2 + self.split_channels3 * 2
        
        # 1x1 and 3x3 convolution layers
        self.conv1x1_1 = nn.Conv2d(self.split_channels1, self.split_channels1, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(self.split_channels2, self.split_channels2, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(self.split_channels3, self.split_channels3, kernel_size=1)
        
        self.conv3x3_1 = nn.Conv2d(self.split_channels1, self.split_channels1, kernel_size=3, padding=1)
        self.conv3x3_2 = nn.Conv2d(self.split_channels2, self.split_channels2, kernel_size=3, padding=1)
        self.conv3x3_3 = nn.Conv2d(self.split_channels3, self.split_channels3, kernel_size=3, padding=1)
        self.conv3x3_4 = nn.Conv2d(self.split_channels3, self.split_channels3, kernel_size=3, padding=1)
        
        # Final 1x1 convolution layer
        self.final_conv1x1 = nn.Conv2d(self.split_channels, in_channels, kernel_size=1)
        
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        distilled_i : i-th distilled feature (preserved)
        coarse_i    : i-th coarse feature (requires further processing)

        Args:
            x: input feature
        Returns:
            id_out: output feature
        """
        
        # First distillation step
        x1, x2      = torch.split(x, self.split_channels1, dim=1)
        distilled_1 = self.conv1x1_1(x1)
        coarse_1    = self.relu(self.conv3x3_1(x2) + x2)
        
        # Second distillation step
        x1, x2      = torch.split(coarse_1, self.split_channels2, dim=1)
        distilled_2 = self.conv1x1_2(x1)
        coarse_2    = self.relu(self.conv3x3_2(x2) + x2)        
        
        # Third distillation step
        x1, x2      = torch.split(coarse_2, self.split_channels3, dim=1)
        distilled_3 = self.conv1x1_3(x1)
        coarse_3    = self.relu(self.conv3x3_3(x2) + x2)        
        
        # Final extraction
        distilled_4 = self.conv3x3_4(coarse_3)
        
        # Concatenate all features
        f_distilled = torch.cat([distilled_1, distilled_2, distilled_3, distilled_4], dim=1)
        # Final 1x1 convolution to fuse the features
        id_out = self.final_conv1x1(f_distilled)
        
        return id_out
    



class MAB(nn.Module):
    """
    Multi-Scale Attention Block.
    
    Args:
        in_chnnels: number of input channels (and output!).
        r: reduction factor of 1x1 convolutions (channel down-sampling an up-sampling).
        n: number of branches.
        kn: lernel size of feature extractor convolutions in each branch (knxkn Conv).
    """

    def __init__(self, in_channels:int, r:int, n:int, kn:list)->None:
        super().__init__()
        
        self.branches = nn.ModuleList()
        for i in range(n):
            kernel_size = kn[i]
            padding = kernel_size // 2
            
            branch = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding),
                nn.Conv2d(in_channels, in_channels // r, kernel_size=1), # WD: channel down-sampling
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // r, in_channels, kernel_size=1), # WU: channel up-sampling
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding)
            )
            
            self.branches.append(branch)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        branch_outputs = []
        
        for branch in self.branches:
            branch_outputs.append(branch(x))
        
        combined_output = sum(branch_outputs)
        sigmoid_output  = torch.sigmoid(combined_output)
        mab_out         = sigmoid_output * x # Element-wise multiplication
        
        return mab_out



class IDMAB(nn.Module):
    # Combination of Information Distillation and Multi-Scale Attention Block.
    def __init__(self, in_channels, param):
        super(IDMAB, self).__init__()
        self.id = ID(in_channels)
        self.mba = MAB(in_channels, param.mba_r, param.mba_n, param.mba_kn)

    def forward(self, x):
        x_in = x
        id_out = self.id(x)
        mba_out = self.mba(id_out)
        return mba_out + x_in
    

class IDMAG(nn.Module):
    """
    Group of IDMAB blocks.
    
    Args:
        param: hyperparameters for IDMAB blocks.
    """
    def __init__(self, param):
        super(IDMAG, self).__init__()
        self.idmab_modules = nn.ModuleList([IDMAB(param.num_features, param) for _ in range(param.B)])
        self.conv = nn.Conv2d(param.num_features, param.num_features, kernel_size=3, padding=1)

    def forward(self, x):
        x_in = x
        for idmab_module in self.idmab_modules:
            x = idmab_module(x)
        x = self.conv(x)
        x = x + x_in
        return x


class DFES(nn.Module):
    """
    Deep Feature Extraction Stage.
    
    Args:
        param: hyperparameters for IDMAG blocks.
    """
    def __init__(self,param):
        super(DFES, self).__init__()
        self.idmag_modules = nn.ModuleList([IDMAG(param) for _ in range(param.G)])
        self.conv = nn.Conv2d(param.num_features, param.num_features, kernel_size=3, padding=1)

    def forward(self, x):
        x_in = x
        for idmag_module in self.idmag_modules:
            x = idmag_module(x)
        x = self.conv(x)
        x = x + x_in
        return x
    
class SFES(nn.Module):
    """
    Shallow Feature Extraction Stage.
    
    Args:
        num_input: number of input channels.
        num_features: number of output channels.
    """
    def __init__(self, num_input, num_features):
        super(SFES, self).__init__()

        self.conv = nn.Conv2d(num_input, num_features, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class PixelShuffle(nn.Module):
    """
    Pixel Shuffle upscaling.
    
    Args:
        input_channel: number of input channels.
        upscale_factor: factor by which to upscale.
    """
    def __init__(self, input_channel , upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor
            # Convolution layer to increase the number of channels by a factor of upscale_factor^2
        self.conv1 = nn.Conv2d(input_channel, input_channel * (upscale_factor**2) , kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Apply the convolution to generate more channels
        x = self.conv1(x)
        batch_size, channels, height, width = x.size()
        r = self.upscale_factor

        # Calculate output dimensions
        output_channels = channels // (r ** 2)
        output_height = height * r
        output_width = width * r

        # Reshape  (batch_size, channels , height, width) -----> channels : output_channels * r * r ----->  (batch_size, output_channels, r, r, height, width)
        x = x.reshape(batch_size, output_channels, r, r, height, width)

        # Permute to (batch_size, output_channels, height, r, width, r)
        x = x.permute(0, 1, 4, 2, 5, 3)

        # Reshape to (batch_size, output_channels, output_height (= height * r), output_width (= width * r))
        x = x.reshape(batch_size, output_channels, output_height, output_width)

        return x
    

class ReconstructionBlock(nn.Module):
    # Reconstruction Block: Applies feedback mechanism followed by convolution
    def __init__(self, in_channels, out_channels):
        super(ReconstructionBlock, self).__init__()
        # Feedback Block
        self.FB = FeedbackBlock( in_channels, out_channels)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # Apply Feedback Block
        x = self.FB(x)
        # Apply final convolution
        x = self.conv(x)
        return x


class FeedbackBlock(nn.Module): 
    # Performs feedback mechanism
    
    def __init__(self, in_channels, out_channels, maxIter=10):
        super(FeedbackBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv  = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # Maximum number of iterations
        self.maxIter = maxIter
    
    def forward(self, x):
        # Store the original input
        all_x = [x]
        x0 = x

        x = self.conv0(x0)
        all_x.append(x)

        # Iterative feedback mechanism
        for i in range(self.maxIter):
          x_pre = x
          x = self.conv(torch.cat([x0, x], dim=1))
          all_x.append(x)
          # Break if change between iterations is below threshold
          if (torch.sum(torch.abs(x - x_pre))) < 1:
            break
        
        # Apply ReLU activation
        x = self.relu(x)

        return x #, all_x

        

# Information Distillation and Multi-Scale Attention Network
class IDMAN(nn.Module):
    def __init__(self, param):
        super(IDMAN, self).__init__()
        # Shallow Feature Extraction Stage
        self.SFES = SFES(param.num_input, param.num_features)
        # Deep Feature Extraction Stage
        self.DFES = DFES(param)
        # Pixel Shuffle for upscaling
        self.UP = PixelShuffle(param.num_features, 2)
        # Reconstruction Block for final output processing
        self.RB = ReconstructionBlock(param.num_features, 1)


    def forward(self, x):
        # Apply shallow feature extraction
        x = self.SFES(x)
        # Apply deep feature extraction
        x = self.DFES(x)
        # Upscale features
        x = self.UP(x)
        # Apply reconstruction block
        x = self.RB(x)
        return x

# Hyperparameters
class params():
    def __init__(self, num_input, num_features, B, G, mba_r, mba_n, mba_kn):
        self.num_input = num_input
        self.num_features = num_features
        self.B = B # Number of IDMAB blocks
        self.G = G # Number of IDMAG blocks
        self.mba_r = mba_r # Reduction factor for Multi-Scale Attention Block
        self.mba_n = mba_n # Reduction factor for Multi-Scale Attention Block
        self.mba_kn = mba_kn # Kernel sizes for each branch in Multi-Scale Attention Block


