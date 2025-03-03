import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedConvEncoder(nn.Module):
    """
    Convolutional encoder with multiple dilation factors.
    Uses dilation factors 2, 4, 6, 8 to capture multi-scale information.
    This replaces the transformer encoder in the original DETR.
    """
    def __init__(self, in_channels=2048, d_model=256):
        super().__init__()
        self.d_model = d_model
        
        # Initial projection to reduce channel dimensions
        self.input_proj = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.bn_input = nn.BatchNorm2d(d_model)
        
        # Dilated convolution blocks with different dilation factors
        self.dilation_blocks = nn.ModuleList([
            self._make_dilated_block(d_model, d_model, dilation=2),
            self._make_dilated_block(d_model, d_model, dilation=4),
            self._make_dilated_block(d_model, d_model, dilation=6),
            self._make_dilated_block(d_model, d_model, dilation=8)
        ])
        
        # Convolutional layer to combine features from different dilation blocks
        self.conv_fusion = nn.Conv2d(d_model * 4, d_model, kernel_size=1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(d_model)
        self.relu = nn.ReLU(inplace=True)
        
        # Layer normalization for the output
        self.layer_norm = nn.LayerNorm(d_model)
        
        self._reset_parameters()
    
    def _make_dilated_block(self, in_channels, out_channels, dilation):
        """Create a dilated convolution block with the specified dilation factor."""
        layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, 
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, 
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return layers
    
    def _reset_parameters(self):
        """Initialize parameters."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, src, mask=None):
        """
        Forward pass for the dilated convolutional encoder.
        Args:
            src: Input features from backbone [batch_size, 2048, H/4, W/4]
            mask: Input padding masks (optional) [batch_size, H, W]
        
        Returns:
            Encoded features [batch_size, HW/16, d_model]
            Updated mask [batch_size, HW/16]
        """
        # Initial projection
        x = self.input_proj(src)
        x = self.bn_input(x)
        x = self.relu(x)
        
        # Process through different dilation blocks and concatenate results
        dilation_outputs = []
        for block in self.dilation_blocks:
            dilation_outputs.append(block(x))
        
        # Concatenate features from different dilation blocks
        x = torch.cat(dilation_outputs, dim=1)
        
        # Fuse features from different dilation blocks
        x = self.conv_fusion(x)
        x = self.bn_fusion(x)
        x = self.relu(x)
        
        # Get dimensions for reshaping
        batch, c, h, w = x.shape
        
        # Update mask if provided
        if mask is not None:
            # Resize mask to match feature map size
            mask = F.interpolate(mask.unsqueeze(1).float(), size=(h, w)).squeeze(1).bool()
            mask = mask.flatten(1)
        
        # Reshape feature map for transformer decoder [B, C, H, W] -> [B, HW, C]
        x = x.flatten(2).permute(0, 2, 1)
        
        # Apply Layer Normalization
        x = self.layer_norm(x)
        
        return x, mask
