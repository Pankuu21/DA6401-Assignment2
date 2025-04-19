import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class FlexibleCNN(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=10,
        conv_filters=[32, 64, 128, 128, 256],
        kernel_sizes=[3, 3, 3, 3, 3],
        activation="relu",
        dense_neurons=512,
        dropout=0.2,
        use_batchnorm=True
    ):
        super().__init__()
        # Store configuration parameters
        self.conv_filters = conv_filters
        self.kernel_sizes = kernel_sizes
        self.dense_neurons = dense_neurons
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        
        # Create model components
        self._build_conv_blocks(activation)
        self._build_classifier(conv_filters[-1] * 7 * 7)

    def _get_activation(self, name):
        """Helper method to get activation function by name"""
        activation_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "mish": nn.Mish(),
        }
        return activation_map.get(name, nn.ReLU())
        
    def _build_conv_blocks(self, activation):
        """Build the convolutional part of the network"""
        self.blocks = nn.ModuleList()
        input_channels = self.in_channels
        
        # Create each convolutional block
        for i, (filters, kernel_size) in enumerate(zip(self.conv_filters, self.kernel_sizes)):
            # Use OrderedDict to maintain layer order
            block_components = OrderedDict()
            
            # Add convolutional layer
            block_components[f'conv{i}'] = nn.Conv2d(
                input_channels, filters, 
                kernel_size=kernel_size,
                padding=kernel_size // 2
            )
            
            # Add batch normalization if specified
            if self.use_batchnorm:
                block_components[f'bn{i}'] = nn.BatchNorm2d(filters)
                
            # Add activation function
            block_components[f'act{i}'] = self._get_activation(activation)
            
            # Add pooling layer
            block_components[f'pool{i}'] = nn.MaxPool2d(2)
            
            # Add dropout if specified
            if self.dropout > 0:
                block_components[f'drop{i}'] = nn.Dropout(self.dropout)
            
            # Create the sequential block and add it
            self.blocks.append(nn.Sequential(block_components))
            input_channels = filters
    
    def _build_classifier(self, flattened_size):
        """Build the classifier part of the network"""
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(flattened_size, self.dense_neurons)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()),
            ('fc2', nn.Linear(self.dense_neurons, self.num_classes))
        ]))

    def forward(self, x):
        # Pass input through convolutional blocks
        for block in self.blocks:
            x = block(x)
        
        # Pass through classifier
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def calculate_parameters_and_computations(self):
        """Calculate model parameters and computational complexity"""
        params_per_layer = []
        computations_per_layer = []
        input_channels = self.in_channels
        
        # Calculate for each convolutional layer
        for i, filters in enumerate(self.conv_filters):
            kernel_size = self.kernel_sizes[i]
            # Parameter count calculation
            param_count = filters * (kernel_size**2 * input_channels + 1)
            
            # Add batch norm parameters if used
            if self.use_batchnorm:
                param_count += 2 * filters  # gamma and beta parameters
            
            # Calculate computational complexity
            output_dim = 224 // (2**(i+1))
            output_size = output_dim**2
            comp_count = filters * kernel_size**2 * input_channels * output_size
            
            params_per_layer.append(param_count)
            computations_per_layer.append(comp_count)
            input_channels = filters
        
        # Calculate dense layer parameters and computations
        last_channels = self.conv_filters[-1]
        dense_param_count = (self.dense_neurons * last_channels * 7 * 7 + 
                            self.dense_neurons +
                            self.num_classes * self.dense_neurons + 
                            self.num_classes)
        
        dense_comp_count = (self.dense_neurons * last_channels * 7 * 7 + 
                           self.num_classes * self.dense_neurons)
        
        return {
            "conv_params": sum(params_per_layer),
            "dense_params": dense_param_count,
            "conv_computations": sum(computations_per_layer),
            "dense_computations": dense_comp_count,
        }

# Example usage
if __name__ == "__main__":
    model = FlexibleCNN(
        conv_filters=[32, 64, 128, 128, 256],
        kernel_sizes=[3, 3, 3, 3, 3],
        dropout=0.2,
        use_batchnorm=True
    )
    results = model.calculate_parameters_and_computations()
    print("Total Parameters:", results["conv_params"] + results["dense_params"])
    print("Total Computations:", results["conv_computations"] + results["dense_computations"])
    
    # Print model architecture
    print(model)
