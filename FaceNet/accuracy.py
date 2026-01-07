import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from utils.import_data import WiderFaceDataset, TRANSFORM
from utils.anchors import AnchorMatcher, AnchorGenerator, box_nms
import numpy as np

VAL_ROOT = r"FaceNet\Dataset\WIDER_val\images"
VAL_ANN_FILE = r"FaceNet\Dataset\wider_ann\wider_face_val_bbx_gt.txt"

class FaceDetectionNet(nn.Module):
    def __init__(self, num_anchors=1):
        super(FaceDetectionNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.det_head = nn.Conv2d(256, num_anchors * 5, kernel_size=1)

    def forward(self, x):
        features = self.backbone(x)
        out = self.det_head(features)
        B, C, H, W = out.shape
        out = out.view(B, -1, 5, H, W)
        return out


def compute_iou_single(box1, box2):
    """
    Compute IoU between two boxes
    box1, box2: [4] (x_min, y_min, x_max, y_max)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h
    
    # Union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    iou = inter_area / (union_area + 1e-8)
    return iou


def decode_predictions(outputs, anchors, conf_threshold=0.5, nms_threshold=0.5):
    """
    Decode network predictions into bounding boxes
    outputs: [batch_size, 1, 5, H, W]
    anchors: [num_anchors, 4]
    Returns: list of predicted boxes per image
    """
    batch_size = outputs.shape[0]
    predictions = []
    
    for i in range(batch_size):
        pred = outputs[i]  # [1, 5, H, W]
        B, C, H, W = pred.shape
        pred_flat = pred.view(B, C, -1).permute(0, 2, 1).contiguous().view(-1, C)  # [H*W, 5]
        
        pred_boxes = pred_flat[:, :4] * 224.0  # Scale back to image coordinates
        pred_conf = torch.sigmoid(pred_flat[:, 4])  # Apply sigmoid to confidence
        
        # Filter by confidence threshold
        conf_mask = pred_conf > conf_threshold
        if conf_mask.sum() == 0:
            predictions.append([])
            continue
        
        filtered_boxes = pred_boxes[conf_mask]
        filtered_conf = pred_conf[conf_mask]
        
        # Apply NMS
        keep_indices = box_nms(filtered_boxes, filtered_conf, nms_threshold)
        final_boxes = filtered_boxes[keep_indices]
        
        predictions.append(final_boxes.cpu().numpy())
    
    return predictions


def evaluate_iou(model, dataloader, anchors, device, conf_threshold=0.5, nms_threshold=0.5):
    """
    Evaluate IoU metrics on validation/test set
    Returns: mean IoU, precision, recall
    """
    model.eval()
    
    all_ious = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    iou_threshold = 0.5  # IoU threshold for considering a detection as correct
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = torch.stack(images).to(device)
            gt_boxes_list = [t['boxes'].cpu().numpy() for t in targets]
            
            # Get predictions
            outputs = model(images)
            predictions = decode_predictions(outputs, anchors, conf_threshold, nms_threshold)
            
            # Compute metrics for each image in batch
            for pred_boxes, gt_boxes in zip(predictions, gt_boxes_list):
                if len(gt_boxes) == 0:
                    false_positives += len(pred_boxes)
                    continue
                
                if len(pred_boxes) == 0:
                    false_negatives += len(gt_boxes)
                    continue
                
                # Match predictions to ground truth boxes
                matched_gt = set()
                
                for pred_box in pred_boxes:
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for gt_idx, gt_box in enumerate(gt_boxes):
                        if gt_idx in matched_gt:
                            continue
                        
                        iou = compute_iou_single(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    if best_iou >= iou_threshold:
                        true_positives += 1
                        matched_gt.add(best_gt_idx)
                        all_ious.append(best_iou)
                    else:
                        false_positives += 1
                
                # Unmatched ground truth boxes are false negatives
                false_negatives += len(gt_boxes) - len(matched_gt)
    
    # Calculate metrics
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'mean_iou': mean_iou,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'num_detections': len(all_ious)
    }


def custom_collate_fn(batch):
    images, targets = zip(*batch)
    return images, targets


if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running validation on: {device}")
    
    # Load test/validation dataset
    # You may need to update this with your validation annotation file
    val_dataset = WiderFaceDataset(
        root_dir=VAL_ROOT,
        annotation_file=VAL_ANN_FILE,
        img_size=224,
        transform=TRANSFORM,
        single_face_only=True
    )
    
    print(f"Validation images: {len(val_dataset)}")
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    # Load model
    model = FaceDetectionNet()
    checkpoint_path = r"FaceNet\saved_checkpoints\faceNet_checkpoint100_multi.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    # Generate anchors (same as training)
    anchor_gen = AnchorGenerator(scales=[1.0], aspect_ratios=[1.0])
    anchors = anchor_gen.generate_anchors(feature_h=28, feature_w=28, stride=8, img_size=224)
    
    # Run evaluation with different thresholds
    print("\n" + "="*60)
    print("Evaluating model performance...")
    print("="*60)
    
    for conf_thresh in [0.3, 0.5, 0.7]:
        print(f"\nConfidence Threshold: {conf_thresh}")
        print("-" * 60)
        
        metrics = evaluate_iou(model, val_dataloader, anchors, device, 
                              conf_threshold=conf_thresh, nms_threshold=0.5)
        
        print(f"Mean IoU:          {metrics['mean_iou']:.4f}")
        print(f"Precision:         {metrics['precision']:.4f}")
        print(f"Recall:            {metrics['recall']:.4f}")
        print(f"F1 Score:          {metrics['f1_score']:.4f}")
        print(f"True Positives:    {metrics['true_positives']}")
        print(f"False Positives:   {metrics['false_positives']}")
        print(f"False Negatives:   {metrics['false_negatives']}")
        print(f"Total Detections:  {metrics['num_detections']}")
    
    print("\n" + "="*60)
    print("Validation complete!")
    print("="*60)

    '''   
    Mean IoU: The average IoU over all bbx over confidence threshold
    Precision: Of the items the model predicted as positive, how many of them were actually positive
    Recall: Of all the actual positive items, how many the model correctly identified
    F1 Score: The harmonic mean of precision and recall
    True Positives: Predictions that were correctly identified
    False Positives: Predictions that were incorrectly identified as positve 
    False Negatives: Cases where the model missed a real positive
    Total Detections: The number of detections the model made correctly above the confidence threshhold
    '''