import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms


class AnchorMatcher:
    """Matches ground truth boxes to anchor boxes using IoU (Intersection over Union)"""
    def __init__(self, pos_iou_thresh=0.5, neg_iou_thresh=0.4):
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
    
    def compute_iou(self, boxes1, boxes2):
        """
        Compute IoU between two sets of boxes
        boxes1: [N, 4] (x_min, y_min, x_max, y_max)
        boxes2: [M, 4]
        Returns: [N, M] IoU matrix
        """
        N, M = boxes1.shape[0], boxes2.shape[0]
        
        # Compute intersection
        inter_xmin = torch.max(boxes1[:, 0:1], boxes2[:, 0])  # [N, M]
        inter_ymin = torch.max(boxes1[:, 1:2], boxes2[:, 1])
        inter_xmax = torch.min(boxes1[:, 2:3], boxes2[:, 2])
        inter_ymax = torch.min(boxes1[:, 3:4], boxes2[:, 3])
        
        inter_w = (inter_xmax - inter_xmin).clamp(min=0)
        inter_h = (inter_ymax - inter_ymin).clamp(min=0)
        inter_area = inter_w * inter_h
        
        # Compute union
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]
        
        union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
        iou = inter_area / (union_area + 1e-8)
        
        return iou
    
    def match(self, anchors, gt_boxes):
        """
        Match ground truth boxes to anchors
        anchors: [num_anchors, 4]
        gt_boxes: [num_gt, 4]
        Returns:
            matched_gt: [num_anchors, 4] - matched gt box for each anchor (or zeros if unmatched)
            labels: [num_anchors] - 1 for positive, 0 for negative, -1 for ignore
        """
        num_anchors = anchors.shape[0]
        
        if gt_boxes.shape[0] == 0:
            # No ground truth - all anchors are negative
            matched_gt = torch.zeros((num_anchors, 4), device=anchors.device, dtype=anchors.dtype)
            labels = torch.zeros(num_anchors, dtype=torch.long, device=anchors.device)
            return matched_gt, labels
        
        iou_matrix = self.compute_iou(anchors, gt_boxes)  # [num_anchors, num_gt]
        
        # For each anchor, find the best matching gt box
        max_iou, matched_gt_idx = iou_matrix.max(dim=1)  # [num_anchors]
        
        matched_gt = gt_boxes[matched_gt_idx]  # [num_anchors, 4]
        
        # Create labels: positive if IoU > threshold, negative if < threshold, ignore otherwise
        labels = torch.full((num_anchors,), -1, dtype=torch.long, device=anchors.device)
        labels[max_iou >= self.pos_iou_thresh] = 1
        labels[max_iou < self.neg_iou_thresh] = 0
        
        return matched_gt, labels


class AnchorGenerator:
    """Generate anchor boxes for the feature map"""
    def __init__(self, scales=[1.0], aspect_ratios=[1.0]):
        self.scales = scales
        self.aspect_ratios = aspect_ratios
    
    def generate_anchors(self, feature_h, feature_w, stride, img_size=224):
        """
        Generate anchors for feature map
        stride: downsampling factor (e.g., 4 for feature map 1/4 of original)
        """
        anchors = []
        anchor_size = 32 * stride  # Base anchor size

        
        for y in range(feature_h):
            for x in range(feature_w):
                cx = (x + 0.5) * stride
                cy = (y + 0.5) * stride
                
                for scale in self.scales:
                    for ratio in self.aspect_ratios:
                        w = anchor_size * scale * (ratio ** 0.5)
                        h = anchor_size * scale / (ratio ** 0.5)
                        
                        x_min = max(0, cx - w / 2)
                        y_min = max(0, cy - h / 2)
                        x_max = min(img_size, cx + w / 2)
                        y_max = min(img_size, cy + h / 2)
                        
                        anchors.append([x_min, y_min, x_max, y_max])
        
        return torch.tensor(anchors, dtype=torch.float32)

def box_nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression
    boxes: [N, 4] (x_min, y_min, x_max, y_max)
    scores: [N] confidence scores
    Returns: indices of boxes to keep
    """
    if boxes.shape[0] == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)
    
    keep = nms(boxes, scores, iou_threshold)
    return keep

def compute_loss_with_anchors(outputs, targets, anchors, matcher, device):
    """
    Compute loss with proper anchor matching
    outputs: [batch_size, 1, 5, H, W] - predictions
    targets: list of [num_gt, 4] ground truth boxes
    anchors: [num_anchors, 4] anchor boxes
    """
    batch_size = outputs.shape[0]
    total_loss = torch.tensor(0.0, device=device)
    valid_samples = 0
    
    for i in range(batch_size):
        gt_boxes = targets[i].to(device)
        
        # Match anchors to ground truth
        matched_boxes, labels = matcher.match(anchors.to(device), gt_boxes)
        
        # Reshape predictions correctly: [1, 5, H, W] -> [H*W, 5]
        pred = outputs[i]  # [1, 5, H, W]
        B, C, H, W = pred.shape
        pred_flat = pred.view(B, C, -1).permute(0, 2, 1).contiguous().view(-1, C)  # [H*W, 5]
        
        pred_boxes = pred_flat[:, :4]
        pred_conf = pred_flat[:, 4]
        
        # Only compute loss for matched anchors
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        if pos_mask.sum() > 0:
            # Regression loss for positive anchors
            pos_pred_boxes = pred_boxes[pos_mask]
            pos_target_boxes = matched_boxes[pos_mask]
            
            reg_loss = F.smooth_l1_loss(pos_pred_boxes, pos_target_boxes / 224.0)
            
            # Confidence loss
            pos_conf_target = torch.ones_like(pred_conf[pos_mask])
            pos_conf_loss = F.binary_cross_entropy_with_logits(pred_conf[pos_mask], pos_conf_target)
            
            # Negative anchor loss
            if neg_mask.sum() > 0:
                neg_conf_target = torch.zeros_like(pred_conf[neg_mask])
                neg_conf_loss = F.binary_cross_entropy_with_logits(pred_conf[neg_mask], neg_conf_target)
            else:
                neg_conf_loss = torch.tensor(0.0, device=device)
            
            total_loss += reg_loss + pos_conf_loss + neg_conf_loss
        else:
            # If no positive anchors, only compute negative loss
            if neg_mask.sum() > 0:
                neg_conf_target = torch.zeros_like(pred_conf[neg_mask])
                neg_conf_loss = F.binary_cross_entropy_with_logits(pred_conf[neg_mask], neg_conf_target)
                total_loss += neg_conf_loss
        
        valid_samples += 1
    
    return total_loss / max(valid_samples, 1)