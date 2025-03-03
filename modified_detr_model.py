import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import FusedBackbone
from dilated_encoder import DilatedConvEncoder
from positional_encoding import PositionalEncoding
from transformer_decoder import TransformerDecoder
from mlp import MLP
from loss import HungarianMatcher, SetCriterion

class HelmetDETR(nn.Module):
    """
    Modified DETR model for helmet detection with 5 classes:
    1. rider_with_helmet
    2. rider_without_helmet
    3. rider_and_passenger_with_helmet
    4. rider_and_passenger_without_helmet
    5. rider_with_helmet_and_passenger_without_helmet
    
    Architecture:
    1. Backbone Network -> 2048 x H/4 x W/4
    2. Dilated Convolutional Encoder (replaces Transformer Encoder)
    3. Transformer Decoder with 20 queries
    4. MLP heads for classification and bounding box regression
    """
    def __init__(self, num_classes=5, backbone_name='resnet50', pretrained=True,
                 d_model=256, nhead=8, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, activation="relu", num_queries=20):
        super().__init__()
        
        # Initialize backbone
        self.backbone = FusedBackbone(name=backbone_name, pretrained=pretrained, dilation=True)
        backbone_output_channels = self.backbone.num_channels
        
        # Initialize dilated convolutional encoder (replaces transformer encoder)
        self.dilated_encoder = DilatedConvEncoder(in_channels=backbone_output_channels, d_model=d_model)
        
        # Initialize positional encoding
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        
        # Initialize transformer decoder
        self.transformer_decoder = TransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            num_queries=num_queries
        )
        
        # Initialize prediction heads
        self.class_embed = nn.Linear(d_model, num_classes + 1)  # +1 for background class
        self.bbox_embed = MLP(d_model, d_model, 4, 3)  # 4 for box coordinates (cx, cy, w, h)
        
        # Initialize parameters
        self._reset_parameters()
        
        # Number of queries
        self.num_queries = num_queries
    
    def _reset_parameters(self):
        """Initialize the weights of the model."""
        # Initialize prediction heads
        nn.init.xavier_uniform_(self.class_embed.weight)
        nn.init.constant_(self.class_embed.bias, 0)
        
        # Initialize bbox MLP
        for layer in self.bbox_embed.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, samples, masks=None):
        """
        Forward pass of the model.
        Args:
            samples: Input images [batch_size, 3, H, W]
            masks: Input padding masks (optional) [batch_size, H, W]
            
        Returns:
            Dict with 'pred_logits' and 'pred_boxes'
        """
        # Extract image features using backbone
        features = self.backbone(samples)  # [batch_size, 2048, H/4, W/4]
        
        # Get image dimensions
        batch_size, _, h, w = features.shape
        
        # Update masks if provided
        if masks is not None:
            # Resize masks to match feature map size
            masks = F.interpolate(masks.unsqueeze(1).float(), size=(h, w)).squeeze(1).bool()
        
        # Pass through dilated convolutional encoder
        memory, masks = self.dilated_encoder(features, masks)  # [batch_size, HW/16, d_model]
        
        # Add positional encoding
        pos_encoding = self.positional_encoding(memory, h, w)
        
        # Pass through transformer decoder
        decoder_output = self.transformer_decoder(memory, masks, pos_encoding)  # [batch_size, num_queries, d_model]
        
        # Get final predictions
        outputs_class = self.class_embed(decoder_output)  # [batch_size, num_queries, num_classes+1]
        outputs_coord = self.bbox_embed(decoder_output).sigmoid()  # [batch_size, num_queries, 4]
        
        # Return predictions
        return {
            'pred_logits': outputs_class,
            'pred_boxes': outputs_coord
        }

    def train_step(self, samples, targets, criterion, optimizer):
        """
        Perform a single training step.
        Args:
            samples: Input images
            targets: Ground truth targets
            criterion: Loss criterion
            optimizer: Optimizer
            
        Returns:
            Dict with losses
        """
        # Forward pass
        outputs = self(samples)
        
        # Compute loss
        loss_dict = criterion(outputs, targets)
        
        # Compute total loss
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        return loss_dict

    @torch.no_grad()
    def inference(self, samples, threshold=0.7):
        """
        Perform inference on input samples.
        Args:
            samples: Input images
            threshold: Score threshold for detection
            
        Returns:
            List of dicts with detections for each image
        """
        # Forward pass
        outputs = self(samples)
        
        # Process each image in the batch
        results = []
        
        # Get predictions
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        # Process predictions for each image
        for img_idx in range(len(pred_logits)):
            # Get scores and labels
            scores, labels = F.softmax(pred_logits[img_idx], dim=-1).max(-1)
            
            # Filter out background class (num_classes) and low confidence detections
            keep = (labels != pred_logits.shape[-1] - 1) & (scores > threshold)
            
            # Get final detections
            boxes = pred_boxes[img_idx, keep]
            scores = scores[keep]
            labels = labels[keep]
            
            # Add to results
            results.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            })
        
        return results


def build_helmet_detr(num_classes=5, pretrained=True, num_queries=20):
    """
    Build the modified DETR model for helmet detection.
    Args:
        num_classes: Number of object classes (5 for helmet detection)
        pretrained: Whether to use pretrained backbone
        num_queries: Number of object queries
        
    Returns:
        model: HelmetDETR model
        criterion: Loss criterion
    """
    # Create model
    model = HelmetDETR(
        num_classes=num_classes,
        backbone_name='resnet50',
        pretrained=pretrained,
        d_model=256,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        num_queries=num_queries
    )
    
    # Create matcher
    matcher = HungarianMatcher(
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0
    )
    
    # Create weight dict
    weight_dict = {
        'loss_ce': 1.0,
        'loss_bbox': 5.0,
        'loss_giou': 2.0
    }
    
    # Create criterion
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=0.1
    )
    
    return model, criterion


# Classes for helmet detection
HELMET_CLASSES = [
    'rider_with_helmet',
    'rider_without_helmet',
    'rider_and_passenger_with_helmet',
    'rider_and_passenger_without_helmet',
    'rider_with_helmet_and_passenger_without_helmet'
]
