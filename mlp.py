import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Multi-layer perceptron for bounding box regression and other tasks.
    Used in the final prediction heads for the modified DETR model.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """
        Initialize the MLP.
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_layers: Number of layers (including output layer)
        """
        super().__init__()
        self.num_layers = num_layers
        
        # Create list of hidden layer dimensions
        h = [hidden_dim] * (num_layers - 1)
        
        # Create layers
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
    
    def forward(self, x):
        """
        Forward pass through the MLP.
        Apply ReLU to all except the final layer.
        Args:
            x: Input tensor [batch_size, ..., input_dim]
            
        Returns:
            Output tensor [batch_size, ..., output_dim]
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
