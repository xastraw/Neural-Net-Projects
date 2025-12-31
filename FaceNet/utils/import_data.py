import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

# This file handles all the loading of the images and their bounding box into a pytorch tensor


# Secures the paths cause im tired of dealing with them
from pathlib import Path
ROOT = Path().resolve().parents[1]
IMG_ROOT = ROOT / "Neural Net Projects" / "FaceNet" / "Dataset" / "WIDER_train" / "images"
ANN_FILE = ROOT / "Neural Net Projects" / "FaceNet" / "Dataset" / "wider_ann" / "wider_face_train_bbx_gt.txt"

#Transform function, resizes the images to 224*224 and scales the bounding boxes accordingly
#Might need to change later on depending on how the net is built
#Define transformations (resize + normalization)
TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

class WiderFaceDataset(Dataset):
    def __init__(self, root_dir, annotation_file, img_size=224, transform=None):
        """
        root_dir: path to WIDER FACE images
        annotation_file: path to annotation file (e.g., 'wider_face_train_bbx_gt.txt')
        img_size: size to resize images
        transform: torchvision transforms for data augmentation
        """

        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform

        # Parse annotation file
        self.data = []
        with open(annotation_file, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            img_path = lines[i].strip()
            num_boxes = int(lines[i+1].strip())

            boxes = []
            if num_boxes > 0:
                for j in range(num_boxes):
                    # Each line: x, y, w, h, blur, expression, illumination, invalid, occlusion, pose
                    bbox_info = list(map(int, lines[i+2+j].strip().split()))
                    x, y, w, h = bbox_info[:4]
                    boxes.append([x, y, x+w, y+h])  # convert to x_min, y_min, x_max, y_max
                i = i + 2 + num_boxes
            else:
                # For when theres a picture with no bounding box, it takes 3 lines so requires extra work
                i +=3

            if len(boxes) == 0: #add pictures with no bounding box to tensor
                boxes = torch.zeros((0, 4), dtype=torch.float32)
            else:
                boxes = torch.tensor(boxes, dtype=torch.float32)

            self.data.append({
                'img_path': os.path.join(root_dir, img_path),
                'boxes': boxes
            })


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = Image.open(sample['img_path']).convert('RGB')
        #boxes = sample['boxes']
        boxes = sample['boxes'].clone()
        #cloning here because if you dont then when upscaling the images again back to their original view you lose the bounding boxes (like for testing purposes)

        #Get original dimensions
        orig_w, orig_h = image.size

        #If there is a transform do that first
        if self.transform:
            image = self.transform(image)
    
        #now scale the boxes based on the original size
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h
        boxes[:, [0,2]] = boxes[:, [0,2]] * scale_x
        boxes[:, [1,3]] = boxes[:, [1,3]] * scale_y

        

        target = {
            'boxes': boxes,
            'num_boxes': boxes.shape[0]
        }
        return image, target

