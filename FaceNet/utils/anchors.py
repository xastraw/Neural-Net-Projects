import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms


class AnchorGeneratorSingle:
    def __init__(self, anchor_sizes=[32, 64, 96]):
        self.anchor_sizes = anchor_sizes

    def generate(self, feature_h, feature_w, stride, device):
        anchors = []
        w = h = self.anchor_sizes

        for size in self.anchor_sizes:
            for y in range(feature_h):
                for x in range(feature_w):
                    cx = (x + 0.5) * stride
                    cy = (y + 0.5) * stride
                    anchors.append([
                        cx - size / 2, cy - size / 2,
                        cx + size / 2, cy + size / 2
                    ])

        return torch.tensor(anchors, device=device, dtype=torch.float32)


def compute_iou(boxes1, boxes2):
    """
    boxes1: [N, 4]
    boxes2:   [1, 4] or [4]
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
    return inter_area / (union + 1e-6) #1e-6 here in case union is 0 (no divison by 0)

def compute_loss_single_face(outputs, targets, anchors, stride):
    """
    outputs: [B, 5, H, W] -> (dx, dy, dw, dh, conf)
    targets: list of dicts with 'boxes'
    anchors: [H*W, 4]
    """

    B, _, H, W = outputs.shape
    outputs = outputs.permute(0, 2, 3, 1).reshape(B, -1, 5)

    #net is prioritizing finding if there is a face not where the face is at, so this should punish that
    stablize_reg = 1.0
    stabalize_conf = 2.0

    total_loss = 0.0

    for i in range(B):
        gt = targets[i]['boxes'][0]  # single face
        gt = gt.to(outputs.device)

        # find best anchor
        ious = compute_iou(anchors, gt.unsqueeze(0))
        best_idx = ious.argmax()

        # anchor box
        ax1, ay1, ax2, ay2 = anchors[best_idx]
        acx = (ax1 + ax2) / 2
        acy = (ay1 + ay2) / 2
        aw = ax2 - ax1
        ah = ay2 - ay1

        #gt is the box that comes from the dataset
        gx1, gy1, gx2, gy2 = gt
        gcx = (gx1 + gx2) / 2
        gcy = (gy1 + gy2) / 2
        gw = gx2 - gx1
        gh = gy2 - gy1

        #targets
        target_offsets = torch.tensor([
            (gcx - acx) / aw,
            (gcy - acy) / ah,
            torch.log(gw / aw),
            torch.log(gh / ah)
        ], device=outputs.device)

        pred_offsets = outputs[i, best_idx, :4]
        pred_conf = outputs[i, best_idx, 4]

        #will penalize anchors with hard negativates
        conf_targets = torch.zeros_like(outputs[i, :, 4])
        conf_targets[best_idx] = 1.0

        # Get confidence scores for all anchors
        all_conf_scores = outputs[i, :, 4].clone()

        # Exclude the positive anchor from negative selection
        all_conf_scores[best_idx] = -float('inf')

        # Select hard negatives (top-k highest confidence false positives)
        # Use 3:1 negative to positive ratio
        num_hard_negs = min(3, len(all_conf_scores) - 1)
        hard_neg_indices = torch.topk(all_conf_scores, num_hard_negs)[1]
        
        # Combine positive anchor with hard negatives
        selected_indices = torch.cat([best_idx.unsqueeze(0), hard_neg_indices])
        
        # Compute confidence loss only for selected anchors
        conf_loss = F.binary_cross_entropy_with_logits(
            outputs[i, selected_indices, 4], 
            conf_targets[selected_indices]
        )
        
        # regression loss
        reg_loss = F.smooth_l1_loss(pred_offsets, target_offsets)

        # combine losses
        total_loss += (stablize_reg * reg_loss) + (stabalize_conf * conf_loss)


    return total_loss / B



