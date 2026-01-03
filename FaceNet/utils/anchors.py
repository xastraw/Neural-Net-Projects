import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms


class AnchorGeneratorSingle:
    def __init__(self, anchor_size=64):
        self.anchor_size = anchor_size

    def generate(self, feature_h, feature_w, stride, device):
        anchors = []
        w = h = self.anchor_size

        for y in range(feature_h):
            for x in range(feature_w):
                cx = (x + 0.5) * stride
                cy = (y + 0.5) * stride
                anchors.append([
                    cx - w / 2, cy - h / 2,
                    cx + w / 2, cy + h / 2
                ])

        return torch.tensor(anchors, device=device, dtype=torch.float32)


def compute_iou(boxes1, boxes2):
    """
    boxes1: [N, 4]
    boxes2: [1, 4] or [4]
    returns: [N]
    """
    if boxes2.dim() == 2:
        boxes2 = boxes2[0]

    inter_x1 = torch.max(boxes1[:, 0], boxes2[0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[3])
    
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])
    
    union = area1 + area2 - inter_area
    return inter_area / (union + 1e-6)

def decode_predictions(pred_offsets, anchors):
    """
    Convert network predictions back to bounding boxes.
    pred_offsets: [N, 4] - (dx, dy, dw, dh)
    anchors: [N, 4] - (x1, y1, x2, y2)
    returns: [N, 4] - predicted boxes in (x1, y1, x2, y2) format
    """
    # Convert anchors to center format
    anchor_cx = (anchors[:, 0] + anchors[:, 2]) / 2
    anchor_cy = (anchors[:, 1] + anchors[:, 3]) / 2
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]
    
    # Decode predictions
    pred_cx = pred_offsets[:, 0] * anchor_w + anchor_cx
    pred_cy = pred_offsets[:, 1] * anchor_h + anchor_cy
    pred_w = torch.exp(pred_offsets[:, 2]) * anchor_w
    pred_h = torch.exp(pred_offsets[:, 3]) * anchor_h
    
    # Convert back to corner format
    pred_boxes = torch.stack([
        pred_cx - pred_w / 2,
        pred_cy - pred_h / 2,
        pred_cx + pred_w / 2,
        pred_cy + pred_h / 2
    ], dim=1)
    
    return pred_boxes



####loss code but meant to penalize badd IoU values
def compute_iou_single(pred_box, gt_box):
    """
    Compute IoU between a single predicted box and ground-truth box.
    pred_box, gt_box: [x1, y1, x2, y2]
    Returns scalar IoU.
    """
    x1 = torch.max(pred_box[0], gt_box[0])
    y1 = torch.max(pred_box[1], gt_box[1])
    x2 = torch.min(pred_box[2], gt_box[2])
    y2 = torch.min(pred_box[3], gt_box[3])

    inter_area = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    union_area = ((pred_box[2]-pred_box[0])*(pred_box[3]-pred_box[1]) +
                  (gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1]) -
                  inter_area + 1e-6)

    iou = inter_area / union_area
    return iou


#Loss code with iou implementation
def compute_loss_single_face(outputs, targets, anchors, stride):
    """
    Loss for single-face detection with IoU penalty.
    outputs: [B, 5, H, W] -> (dx, dy, dw, dh, conf)
    targets: list of dicts with 'boxes' tensor [x1, y1, x2, y2]
    anchors: [num_anchors, 4] -> absolute coordinates
    """
    B, _, H, W = outputs.shape
    outputs = outputs.permute(0, 2, 3, 1).reshape(B, -1, 5)

    weight_reg = 5.0
    weight_conf = 1.0
    weight_iou = 2.0  # IoU penalty weight

    total_loss = 0.0
    total_reg_loss = 0.0
    total_conf_loss = 0.0
    total_iou_loss = 0.0

    for i in range(B):
        gt = targets[i]['boxes'][0].to(outputs.device)  # single face

        # Compute IoUs with all anchors
        ious = compute_iou(anchors, gt.unsqueeze(0))  # [num_anchors]
        best_idx = ious.argmax()
        best_iou = ious[best_idx]

        # Anchor box
        ax1, ay1, ax2, ay2 = anchors[best_idx]
        acx = (ax1 + ax2) / 2
        acy = (ay1 + ay2) / 2
        aw = ax2 - ax1
        ah = ay2 - ay1

        # GT box
        gx1, gy1, gx2, gy2 = gt
        gcx = (gx1 + gx2) / 2
        gcy = (gy1 + gy2) / 2
        gw = gx2 - gx1
        gh = gy2 - gy1

        # Offsets
        target_offsets = torch.tensor([
            (gcx - acx) / aw,
            (gcy - acy) / ah,
            torch.log(gw / aw),
            torch.log(gh / ah)
        ], device=outputs.device)
        target_offsets = torch.clamp(target_offsets, -5.0, 5.0)

        pred_offsets = outputs[i, best_idx, :4]

        # Regression loss (Smooth L1)
        reg_loss = F.smooth_l1_loss(pred_offsets, target_offsets)

        # Convert predicted offsets back to absolute box coords for IoU
        px = pred_offsets[0] * aw + acx
        py = pred_offsets[1] * ah + acy
        pw = torch.exp(pred_offsets[2]) * aw
        ph = torch.exp(pred_offsets[3]) * ah

        pred_box = torch.stack([px - pw/2, py - ph/2, px + pw/2, py + ph/2])
        iou = compute_iou_single(pred_box, gt)
        iou_loss_val = 1.0 - iou  # penalize low IoU

        # Confidence targets
        conf_targets = torch.zeros_like(outputs[i, :, 4])
        conf_targets[best_idx] = 1.0

        # Hard negative mining
        all_conf_scores = outputs[i, :, 4].clone()
        all_conf_scores[best_idx] = -float('inf')
        num_hard_negs = min(5, len(all_conf_scores) - 1)
        hard_neg_indices = torch.topk(all_conf_scores, num_hard_negs)[1]
        selected_indices = torch.cat([best_idx.unsqueeze(0), hard_neg_indices])

        conf_loss = F.binary_cross_entropy_with_logits(
            outputs[i, selected_indices, 4],
            conf_targets[selected_indices]
        )

        # Total per-image loss
        total_loss += weight_reg*reg_loss + weight_conf*conf_loss + weight_iou*iou_loss_val
        total_reg_loss += reg_loss.item()
        total_conf_loss += conf_loss.item()
        total_iou_loss += iou_loss_val.item()

    avg_reg = total_reg_loss / B
    avg_conf = total_conf_loss / B
    avg_iou = total_iou_loss / B

    return total_loss / B, avg_reg, avg_conf, avg_iou



#original loss code
def compute_loss_single_face1(outputs, targets, anchors, stride):
    """
    IMPROVED loss function with better debugging and balance.
    outputs: [B, 5, H, W] -> (dx, dy, dw, dh, conf)
    targets: list of dicts with 'boxes'
    anchors: [H*W, 4]
    """
    B, _, H, W = outputs.shape
    outputs = outputs.permute(0, 2, 3, 1).reshape(B, -1, 5)

    # CHANGED: More balanced weights
    weight_reg = 5.0      # Focus on getting boxes right
    weight_conf = 1.0     # But don't ignore confidence

    total_loss = 0.0
    total_reg_loss = 0.0
    total_conf_loss = 0.0

    for i in range(B):
        gt = targets[i]['boxes'][0]  # single face
        gt = gt.to(outputs.device)

        # Find best anchor based on IoU
        ious = compute_iou(anchors, gt.unsqueeze(0))
        best_idx = ious.argmax()
        best_iou = ious[best_idx]

        # DEBUGGING: Check if its matching reasonable anchors
        # Uncomment to debug:
        # if i == 0:
        #     print(f"Best IoU: {best_iou:.4f}, GT box: {gt}")

        # Get anchor parameters
        ax1, ay1, ax2, ay2 = anchors[best_idx]
        acx = (ax1 + ax2) / 2
        acy = (ay1 + ay2) / 2
        aw = ax2 - ax1
        ah = ay2 - ay1

        # Get ground truth parameters
        gx1, gy1, gx2, gy2 = gt
        gcx = (gx1 + gx2) / 2
        gcy = (gy1 + gy2) / 2
        gw = gx2 - gx1
        gh = gy2 - gy1

        # CHANGED: Add epsilon to prevent log(0)
        gw = torch.clamp(gw, min=1.0)
        gh = torch.clamp(gh, min=1.0)
        aw = torch.clamp(aw, min=1.0)
        ah = torch.clamp(ah, min=1.0)

        # Compute target offsets
        target_offsets = torch.tensor([
            (gcx - acx) / aw,
            (gcy - acy) / ah,
            torch.log(gw / aw),
            torch.log(gh / ah)
        ], device=outputs.device)

        # CHANGED: Clamp extreme values
        target_offsets = torch.clamp(target_offsets, min=-5.0, max=5.0)

        pred_offsets = outputs[i, best_idx, :4]

        # === HARD NEGATIVE MINING ===
        conf_targets = torch.zeros_like(outputs[i, :, 4])
        conf_targets[best_idx] = 1.0
        
        # Select hard negatives
        all_conf_scores = outputs[i, :, 4].clone()
        all_conf_scores[best_idx] = -float('inf')
        
        # CHANGED: Use 5:1 ratio for more negative examples
        num_hard_negs = min(5, len(all_conf_scores) - 1)
        hard_neg_indices = torch.topk(all_conf_scores, num_hard_negs)[1]
        
        selected_indices = torch.cat([best_idx.unsqueeze(0), hard_neg_indices])
        
        conf_loss = F.binary_cross_entropy_with_logits(
            outputs[i, selected_indices, 4], 
            conf_targets[selected_indices]
        )
        
        # Regression loss (only for positive anchor)
        reg_loss = F.smooth_l1_loss(pred_offsets, target_offsets)
        
        total_reg_loss += reg_loss.item()
        total_conf_loss += conf_loss.item()
        
        # Combine losses
        total_loss += (weight_reg * reg_loss) + (weight_conf * conf_loss)

    # Print average losses for debugging
    avg_reg = total_reg_loss / B
    avg_conf = total_conf_loss / B
    
    return total_loss / B, avg_reg, avg_conf



