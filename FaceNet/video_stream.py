import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from utils.import_data import WiderFaceDataset, TRANSFORM, TRAIN_ROOT, TRAIN_ANN_FILE, TEST_ROOT
from utils.anchors import AnchorMatcher, AnchorGenerator, box_nms, compute_loss_with_anchors

class FaceDetectionNet(nn.Module):
    def __init__(self, num_anchors=1):
        """
        num_anchors: number of boxes predicted per spatial cell (simplest: 1)
        """
        super(FaceDetectionNet, self).__init__()

        """
        kernel_size is the size of the box we pass over each img to extract the features, exactly like tf (3,3,3)
        """
        #Backbone (feature extractor)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # RGB input
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # downsample by 2 -> 112x1112

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # downsample by 2 -> 56x56

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # downsample by 2 -> 28x28
        )

        # Detection head
        # Predict bounding boxes + confidence
        # Output channels = num_anchors * 5 (x, y, w, h, conf)
        self.det_head = nn.Conv2d(256, num_anchors * 5, kernel_size=1)

    def forward(self, x):
        """
        x: [batch_size, 3, H, W]
        Returns:
            out: [batch_size, num_anchors * 5, H/4, W/4] 
                 Each cell predicts (x, y, w, h, confidence)
        """
        features = self.backbone(x)
        out = self.det_head(features)  # [B, 5*num_anchors, H', W']

        B, C, H, W = out.shape
        out = out.view(B, -1, 5, H, W)  # [B, num_anchors, 5, H', W']
        return out        

def preprocess(frame, size=(224, 224)):
    # OpenCV frame is BGR uint8 (H, W, 3)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(rgb, size)               # (H, W, 3)
    x = resized.astype(np.float32) / 255.0        # scale 0..1

    # HWC -> CHW
    x = np.transpose(x, (2, 0, 1))                # (3, H, W)

    # Add batch dimension: (3,H,W)->(1,3,H,W)
    x = np.expand_dims(x, axis=0)

    # NumPy -> Torch tensor
    x = torch.from_numpy(x).to(device)            # float32 tensor on device
    return x

def get_boxes(image):
    #load image w/ preprocessing
    
    orig_w, orig_h = image.size  # save original size

    image_tensor = TRANSFORM(image)   # same TRANSFORM as training
    image_tensor = image_tensor.unsqueeze(0).to(device)  # [1, 3, 224, 224]

    #forward pass w/ no gradient
    with torch.no_grad():
        outputs = model(image_tensor)

    #decode the predicitions into boxes and scores
    pred = outputs[0]  # [1, 5, 28, 28]

    # Flatten
    pred = pred.view(5, -1).permute(1, 0)  # [28*28, 5]

    pred_boxes = pred[:, :4] * 224.0   # undo normalization
    pred_scores = torch.sigmoid(pred[:, 4])

    #apply confidence threshhold and nms (non max suppresion)
    CONF_THRESH = 0.5
    NMS_THRESH = 0.4

    keep = pred_scores > CONF_THRESH
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]

    if pred_boxes.shape[0] > 0:
        keep_idx = box_nms(pred_boxes, pred_scores, iou_threshold=NMS_THRESH)
        pred_boxes = pred_boxes[keep_idx]
        pred_scores = pred_scores[keep_idx]


    #scale boxes back into original image
    scale_x = orig_w / 224
    scale_y = orig_h / 224

    pred_boxes[:, [0, 2]] *= scale_x
    pred_boxes[:, [1, 3]] *= scale_y

    return pred_boxes, pred_scores

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    frame_skip = 5
    frame_count = 0

    cached_boxes = None
    cached_scores = None

    no_face_frames = 0
    cache_clear_interval = 5
    max_no_face_frames = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % frame_skip == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pred_boxes, pred_scores = get_boxes(pil_img)

            if pred_boxes is None or len(pred_boxes) > 0:
                cached_boxes = pred_boxes
                cached_scores = pred_scores
                no_face_frames = 0
            else:
                no_face_frames += 1
        
        if no_face_frames >= cache_clear_interval:
            cached_boxes = None
            cached_scores = None

        if no_face_frames >= max_no_face_frames:
            print("No faces detected for a while, Exiting.")
            break

        if cached_boxes is not None:
            for box, score in zip(cached_boxes, cached_scores):
                x1, y1, x2, y2 = box.cpu()
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f"Confidence: {score:.2f}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("Face Detector", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

        if cv2.getWindowProperty("Face Detector", cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #set device using the gpu if it is available otherwise use cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #Load model and weights
    model = FaceDetectionNet()
    checkpoint_path = "FaceNet/saved_checkpoints/faceNet_checkpoint100_multi.pth"
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()    
    main()