import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms as T
from utils.anchors import AnchorGeneratorSingle, decode_predictions

# Define the network architecture (must match your training)

class FaceDetectionNet(nn.Module):
    def __init__(self, num_anchors=1):
        """
        num_anchors: the number of anchor sizes (3: [32, 64, 96])
        kernel_size is the size of the box we pass over each img to extract the features, exactly like tf (3,3,3)
        """
        super(FaceDetectionNet, self).__init__()

        #Backbone (feature extractor)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # RGB input
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Detection head
        # Predict bounding boxes + confidence
        # Output channels = num_anchors * 5 (x, y, w, h, conf)
        self.det_head = nn.Conv2d(128, num_anchors * 5, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        return self.det_head(x)


# Image preprocessing transform (must match training)
TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model(checkpoint_path, num_anchors=2, device='cuda'):
    """
    Load the trained model from checkpoint.
    """
    # Create model
    net = FaceDetectionNet(num_anchors=num_anchors)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint["model_state"])
    
    # Move to device and set to eval mode
    net = net.to(device)
    net.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch'] + 1} epochs")
    
    return net


def predict_face(net, image_path, anchors, device='cuda', conf_thresh=0.3):
    """
    Predict face location in an image.
    
    Returns:
        pred_box: [x1, y1, x2, y2] in original image coordinates
        confidence: confidence score
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    orig_w, orig_h = image.size
    
    image_tensor = TRANSFORM(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)  # [1, 3, 224, 224]
    
    # Forward pass
    with torch.no_grad():
        outputs = net(image_tensor)  # [1, num_anchors*5, 28, 28]
    
    # Reshape outputs
    B, C, H, W = outputs.shape
    num_anchors = C // 5
    
    if num_anchors == 1:
        outputs = outputs.permute(0, 2, 3, 1).reshape(B, -1, 5)
    else:
        outputs = outputs.view(B, num_anchors, 5, H, W)
        outputs = outputs.permute(0, 3, 4, 1, 2)
        outputs = outputs.reshape(B, -1, 5)
    
    # Get predictions
    pred = outputs[0]  # [H*W*num_anchors, 5]
    pred_offsets = pred[:, :4]
    pred_scores = torch.sigmoid(pred[:, 4])
    
    # Decode to actual boxes
    pred_boxes = decode_predictions(pred_offsets, anchors)
    
    # Filter by confidence
    keep = pred_scores > conf_thresh
    if keep.sum() == 0:
        print(f"No detections above confidence threshold {conf_thresh}")
        return None, 0.0
    
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]
    
    # Get highest confidence detection
    best_idx = pred_scores.argmax()
    pred_box = pred_boxes[best_idx].cpu()
    confidence = pred_scores[best_idx].cpu().item()
    
    # Scale back to original image size
    scale_x = orig_w / 224
    scale_y = orig_h / 224
    
    pred_box[0] *= scale_x  # x1
    pred_box[2] *= scale_x  # x2
    pred_box[1] *= scale_y  # y1
    pred_box[3] *= scale_y  # y2
    
    return pred_box, confidence


def visualize_detection(image_path, pred_box, confidence, save_path=None):
    """
    Visualize the detected face with bounding box.
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    
    if pred_box is not None:
        # Draw bounding box
        x1, y1, x2, y2 = pred_box
        width = x2 - x1
        height = y2 - y1
        
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=3, edgecolor='blue', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add confidence text
        ax.text(
            x1, y1 - 10,
            f'Confidence: {confidence:.2f}',
            color='red',
            fontsize=12,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7)
        )
    else:
        ax.text(
            10, 30,
            'No face detected',
            color='red',
            fontsize=14,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7)
        )
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def test_on_multiple_images(net, image_folder, anchors, device='cuda', conf_thresh=0.3):
    """
    Test the network on multiple images in a folder.
    """
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(image_folder) 
                   if os.path.splitext(f)[1].lower() in image_extensions]
    
    print(f"Found {len(image_files)} images in {image_folder}")
    
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        print(f"\nProcessing {img_file}...")
        
        # Predict
        pred_box, confidence = predict_face(net, img_path, anchors, device, conf_thresh)
        
        # Visualize
        visualize_detection(img_path, pred_box, confidence)


# ==================== MAIN USAGE ====================

if __name__ == "__main__":
    # Configuration
    #C:\Code\Neural Net Projects\FaceNet\saved_checkpoints\faceNet_checkpoint60.pth
    #checkpoint_path =

    #CHECKPOINT_PATH = os.path.join("saved_checkpoints", "faceNet_checkpoint60.pth")
    CHECKPOINT_PATH = r"C:\Code\Neural Net Projects\FaceNet\saved_checkpoints\faceNet_checkpoint60.pth"
    IMAGE_PATH= r"C:\\Code\\Neural Net Projects\\FaceNet\\Dataset\\WIDER_test\\images\\0--Parade\\0_Parade_Parade_0_1046.jpg"
    NUM_ANCHORS = 1  # Must match = training
    ANCHOR_SIZES = 42  # Must match your training
    CONF_THRESHOLD = 0.1  # Lower = more detections, higher = fewer but more confident
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate anchors (must match training)
    anchor_gen = AnchorGeneratorSingle()
    anchors = anchor_gen.generate(feature_h=28, feature_w=28, stride=8, device=device)
    print(f"Generated {anchors.shape[0]} anchors")
    
    # Load model
    net = load_model(CHECKPOINT_PATH, num_anchors=NUM_ANCHORS, device=device)
    
    # Test on single image
    print(f"\nTesting on {IMAGE_PATH}...")
    pred_box, confidence = predict_face(net, IMAGE_PATH, anchors, device, CONF_THRESHOLD)
    
    if pred_box is not None:
        print(f"Detection: Box=[{pred_box[0]:.1f}, {pred_box[1]:.1f}, {pred_box[2]:.1f}, {pred_box[3]:.1f}], Confidence={confidence:.2f}")
    else:
        print("No face detected")
    
    # Visualize
    visualize_detection(IMAGE_PATH, pred_box, confidence, save_path="detection_result.jpg")
    
    # Optional: Test on multiple images
    # test_on_multiple_images(net, "test_images/", anchors, device, CONF_THRESHOLD)