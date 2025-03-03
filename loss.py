import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

class FocalLoss(nn.Module):
    """
    Focal Loss for classification to address class imbalance.
    Optimized for helmet detection with 5 classes.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Calculate focal loss.
        Args:
            inputs: Predicted class logits [B, num_classes, N]
            targets: Target classes [B, N]
        
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)
        loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class GIoULoss(nn.Module):
    """
    Generalized IoU loss for better bounding box regression.
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, preds, targets):
        """
        Calculate GIoU loss.
        Args:
            preds: Predicted boxes [N, 4] in format (cx, cy, w, h) normalized
            targets: Target boxes [N, 4] in format (cx, cy, w, h) normalized
        
        Returns:
            GIoU loss
        """
        # Convert from center-width format to min-max format
        pred_boxes = self._box_cxcywh_to_xyxy(preds)
        target_boxes = self._box_cxcywh_to_xyxy(targets)
        
        # Calculate intersection
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        # Handle non-intersection cases
        inter_width = (inter_x2 - inter_x1).clamp(min=0)
        inter_height = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_width * inter_height
        
        # Calculate areas of boxes
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        
        # Calculate union area
        union_area = pred_area + target_area - inter_area
        
        # Calculate IoU
        iou = inter_area / (union_area + 1e-7)
        
        # Calculate smallest enclosing box
        enclosing_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclosing_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclosing_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclosing_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        
        # Calculate area of enclosing box
        enclosing_width = enclosing_x2 - enclosing_x1
        enclosing_height = enclosing_y2 - enclosing_y1
        enclosing_area = enclosing_width * enclosing_height
        
        # Calculate GIoU
        giou = iou - (enclosing_area - union_area) / (enclosing_area + 1e-7)
        loss = 1 - giou
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def _box_cxcywh_to_xyxy(self, x):
        """
        Convert bounding boxes from (cx, cy, w, h) to (x1, y1, x2, y2).
        """
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)


class HungarianMatcher(nn.Module):
    """
    Assign predictions to ground truth using the Hungarian algorithm.
    """
    def __init__(self, cost_class=1.0, cost_bbox=1.0, cost_giou=1.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.giou_loss = GIoULoss(reduction='none')
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Match predictions to ground truth.
        Args:
            outputs: Dict with 'pred_logits' and 'pred_boxes'
            targets: List of dicts with 'labels' and 'boxes'
        
        Returns:
            List of tuples (pred_indices, gt_indices) for each batch element
        """
        batch_size, num_queries = outputs['pred_logits'].shape[:2]
        
        # Flatten predictions
        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs['pred_boxes'].flatten(0, 1)  # [batch_size * num_queries, 4]
        
        # List to store indices
        indices = []
        
        # Process each batch element separately
        for b in range(batch_size):
            # Get ground truth for this batch element
            tgt_ids = targets[b]['labels']
            tgt_bbox = targets[b]['boxes']
            
            if tgt_ids.shape[0] == 0:
                # No ground truth boxes, assign dummy matching
                indices.append((torch.arange(0, 0, device=out_prob.device),
                               torch.arange(0, 0, device=out_prob.device)))
                continue
            
            # Classification cost: -log probability of the correct class
            cost_class = -out_prob[b * num_queries:(b + 1) * num_queries, tgt_ids]
            
            # Bounding box L1 cost
            cost_bbox = torch.cdist(
                out_bbox[b * num_queries:(b + 1) * num_queries], 
                tgt_bbox, 
                p=1
            )
            
            # GIoU cost
            cost_giou = self.giou_loss(
                out_bbox[b * num_queries:(b + 1) * num_queries].unsqueeze(1).expand(-1, tgt_ids.shape[0], -1),
                tgt_bbox.unsqueeze(0).expand(num_queries, -1, -1)
            ).squeeze(-1)
            
            # Combine costs
            C = (
                self.cost_class * cost_class + 
                self.cost_bbox * cost_bbox + 
                self.cost_giou * cost_giou
            )
            
            # Use Hungarian algorithm to find optimal assignment
            C = C.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(C)
            
            # Convert to tensors and append to indices
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.int64, device=out_prob.device),
                torch.as_tensor(col_ind, dtype=torch.int64, device=out_prob.device)
            ))
        
        return [(torch.tensor(i, dtype=torch.int64, device=out_prob.device) + b * num_queries, j) 
                for b, (i, j) in enumerate(indices)]


class SetCriterion(nn.Module):
    """
    Loss computation for Modified DETR.
    Uses Focal Loss for classification and combination of L1 and GIoU losses for boxes.
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef  # Weight for no-object class
        self.focal_loss = FocalLoss()
        self.l1_loss = nn.L1Loss()
        self.giou_loss = GIoULoss()
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef  # Last class is no-object
        self.register_buffer('empty_weight', empty_weight)
    
    def forward(self, outputs, targets):
        """
        Compute the loss for the helmet detection task.
        Args:
            outputs: Dict with 'pred_logits' and 'pred_boxes'
            targets: List of dicts with 'labels' and 'boxes'
        
        Returns:
            Dict of losses
        """
        # Get indices from matcher
        indices = self.matcher(outputs, targets)
        
        # Compute classification loss
        loss_ce = self._classification_loss(outputs, targets, indices)
        
        # Compute box loss
        loss_bbox, loss_giou = self._box_loss(outputs, targets, indices)
        
        # Return losses
        losses = {
            'loss_ce': loss_ce,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou
        }
        
        # Scale losses according to weight_dict
        return {k: v * self.weight_dict[k] for k, v in losses.items() if k in self.weight_dict}
    
    def _classification_loss(self, outputs, targets, indices):
        """
        Compute the focal loss for classification.
        """
        pred_logits = outputs['pred_logits']
        batch_size, num_queries = pred_logits.shape[:2]
        
        # Create target tensor with default class = no-object (num_classes)
        target_classes = torch.full(
            (batch_size, num_queries), 
            self.num_classes,
            dtype=torch.int64, 
            device=pred_logits.device
        )
        
        # Fill in the matched targets
        for batch_idx in range(batch_size):
            if indices[batch_idx][0].numel() > 0:  # If there are any matches
                target_classes[batch_idx, indices[batch_idx][0]] = targets[batch_idx]['labels'][indices[batch_idx][1]]
        
        # Apply focal loss
        return self.focal_loss(pred_logits.transpose(1, 2), target_classes)
    
    def _box_loss(self, outputs, targets, indices):
        """
        Compute the L1 and GIoU losses for bounding boxes.
        """
        pred_boxes = outputs['pred_boxes']
        
        # Get the matched pairs
        src_boxes = []
        target_boxes = []
        
        for batch_idx in range(len(targets)):
            if indices[batch_idx][0].numel() > 0:  # If there are any matches
                src_idx = indices[batch_idx][0]
                tgt_idx = indices[batch_idx][1]
                src_boxes.append(pred_boxes[batch_idx, src_idx])
                target_boxes.append(targets[batch_idx]['boxes'][tgt_idx])
        
        if not src_boxes:  # No matching pairs
            return torch.tensor(0.0, device=pred_boxes.device), torch.tensor(0.0, device=pred_boxes.device)
        
        # Stack tensors
        src_boxes = torch.cat(src_boxes)
        target_boxes = torch.cat(target_boxes)
        
        # Compute L1 loss
        loss_bbox = self.l1_loss(src_boxes, target_boxes)
        
        # Compute GIoU loss
        loss_giou = self.giou_loss(src_boxes, target_boxes)
        
        return loss_bbox, loss_giou
