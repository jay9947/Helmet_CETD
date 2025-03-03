import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Fixed positional encodings for 2D feature maps.
    Uses sine and cosine functions of different frequencies.
    """
    def __init__(self, d_model, max_h=50, max_w=50, temperature=10000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding buffer
        pe = torch.zeros(max_h, max_w, d_model)
        
        # Create positional encodings for 2D grid
        y_pos = torch.arange(0, max_h).unsqueeze(1).expand(-1, max_w).float()
        x_pos = torch.arange(0, max_w).expand(max_h, -1).float()
        
        # Calculate position encoding for each dimension
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(temperature) / d_model))
        
        # Apply sin to even indices and cos to odd indices
        pe[:, :, 0::2] = torch.sin(x_pos.unsqueeze(-1) * div_term)
        pe[:, :, 1::2] = torch.cos(x_pos.unsqueeze(-1) * div_term)
        
        # Add the y-dimension positional encoding
        y_pe = torch.zeros(max_h, max_w, d_model)
        y_pe[:, :, 0::2] = torch.sin(y_pos.unsqueeze(-1) * div_term)
        y_pe[:, :, 1::2] = torch.cos(y_pos.unsqueeze(-1) * div_term)
        
        # Combine x and y positional encodings
        pe = pe + y_pe
        
        # Register positional encoding as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x, h, w):
        """
        Get positional encodings for a feature map of size h×w.
        Args:
            x: Input tensor of shape [batch_size, h*w, d_model]
            h: Height of the feature map
            w: Width of the feature map
        
        Returns:
            Positional encodings of shape [batch_size, h*w, d_model]
        """
        # Extract appropriate positional encodings for the current resolution
        pos_enc = self.pe[:h, :w, :].reshape(h * w, self.d_model)
        
        # Expand to batch size
        return pos_enc.unsqueeze(0).expand(x.size(0), -1, -1)
