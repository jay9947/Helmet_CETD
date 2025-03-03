import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class FusedBackbone(nn.Module):
    """
    Fused backbone network for feature extraction.
    Takes an input image and produces a feature map of shape (batch_size, 2048, H/4, W/4)
    """
    def __init__(self, name='resnet50', pretrained=True, dilation=False):
        super().__init__()
        
        # Get the specified backbone network
        if name == 'resnet50':
            backbone = torchvision.models.resnet50(pretrained=pretrained)
        elif name == 'resnet101':
            backbone = torchvision.models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {name}")
        
        # Use only the layers up to layer3
        self.backbone_layers = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3
        )
        
        # Use layer4 with dilation if specified
        if dilation:
            # Replace stride with dilation in layer4
            for m in backbone.layer4.modules():
                if isinstance(m, nn.Conv2d):
                    if m.stride == (2, 2):
                        m.stride = (1, 1)
                        m.dilation = (2, 2)
                        m.padding = (2, 2)
            self.layer4 = backbone.layer4
        else:
            self.layer4 = backbone.layer4
            
        # Set feature dimensions
        self.num_channels = 2048  # ResNet output channels
        
        # Freeze backbone parameters if pretrained
        if pretrained:
            for param in self.backbone_layers.parameters():
                param.requires_grad = False
                
    def forward(self, x):
        """
        Forward pass through the backbone.
        Args:
            x: Input tensor of shape [batch_size, 3, H, W]
            
        Returns:
            Feature map of shape [batch_size, 2048, H/4, W/4]
        """
        # Get features from the backbone
        features = self.backbone_layers(x)
        
        # Get features from layer4
        features = self.layer4(features)
        
        return features
