import os
import logging
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image

# ==========================================
# 0. CONFIGURATION & LOGGING
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger("RetinaNet-TransferLearning")

CONFIG = {
    "num_classes": 2, # 1 class (Face) + 1 background class (Mandatory for PyTorch)
    "batch_size": 4,  # Object detection requires more VRAM, keep batch size small
    "epochs": 5,
    "learning_rate": 1e-4,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "num_mock_images": 20
}

# ==========================================
# 1. MODEL ARCHITECTURE (TRANSFER LEARNING)
# ==========================================
def get_face_detection_model(num_classes: int, freeze_backbone: bool = True):
    """
    Loads a pre-trained RetinaNet, freezes its feature extractor, 
    and replaces the classification head for Face Detection.
    """
    logger.info("Loading pre-trained RetinaNet-ResNet50-FPN (COCO)...")
    # Load a model pre-trained on COCO (91 classes)
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    
    if freeze_backbone:
        logger.info("Freezing ResNet50 Backbone and FPN layers...")
        for param in model.backbone.parameters():
            param.requires_grad = False

    logger.info(f"Replacing classification head for {num_classes} classes (Background + Face)...")
    # RetinaNet has a specific classification head structure per anchor point
    num_anchors = model.head.classification_head.num_anchors
    in_channels = model.backbone.out_channels
    
    # Replace the existing head with a new one adapted for our specific number of classes
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels, 
        num_anchors=num_anchors, 
        num_classes=num_classes
    )
    
    # We leave the regression head (which predicts bounding box coordinates) intact, 
    # as it already knows how to draw boxes. We just fine-tune it during training.
    
    return model

# ==========================================
# 2. DATASET PREPARATION & MOCKING
# ==========================================
class MockFaceDetectionDataset(Dataset):
    """
    A custom PyTorch Dataset for Object Detection.
    Generates random images and random valid bounding boxes to simulate face annotations.
    """
    def __init__(self, num_images: int):
        self.num_images = num_images
        self.img_size = 300 # 300x300 images
        logger.info(f"Initialized Mock Dataset with {num_images} samples.")

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # 1. Create a mock image (random noise)
        img_array = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_tensor = F.to_tensor(img) # Converts to [C, H, W] and normalizes [0.0, 1.0]

        # 2. Create mock bounding boxes [x_min, y_min, x_max, y_max]
        # Let's simulate 1 to 3 faces per image
        num_faces = np.random.randint(1, 4)
        boxes = []
        labels = []
        
        for _ in range(num_faces):
            x_min = np.random.randint(0, self.img_size - 50)
            y_min = np.random.randint(0, self.img_size - 50)
            x_max = np.random.randint(x_min + 20, self.img_size)
            y_max = np.random.randint(y_min + 20, self.img_size)
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(1) # Class 1 is 'Face' (0 is background)

        # Convert to PyTorch tensors
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        
        # Area is required by PyTorch evaluators
        area = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])
        target["area"] = area
        target["iscrowd"] = torch.zeros((num_faces,), dtype=torch.int64) # Assume no crowds

        return img_tensor, target

def collate_fn(batch):
    """
    Custom collate function is REQUIRED for object detection datasets
    because images have varying numbers of bounding boxes.
    """
    return tuple(zip(*batch))

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_detector():
    logger.info(f"Using device: {CONFIG['device']}")
    
    # 1. Prep Data
    dataset = MockFaceDetectionDataset(num_images=CONFIG["num_mock_images"])
    dataloader = DataLoader(
        dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        collate_fn=collate_fn
    )

    # 2. Init Model & Optimizer
    model = get_face_detection_model(num_classes=CONFIG["num_classes"], freeze_backbone=True)
    model.to(CONFIG["device"])
    
    # Optimize only the parameters that require gradients (the new head + unfreezed layers)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=CONFIG["learning_rate"])

    # 3. Training Loop
    logger.info("Starting RetinaNet Transfer Learning...")
    for epoch in range(CONFIG["epochs"]):
        model.train() # In train mode, torchvision detection models return losses automatically
        epoch_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            # Move data to GPU/CPU
            images = list(image.to(CONFIG["device"]) for image in images)
            targets = [{k: v.to(CONFIG["device"]) for k, v in t.items()} for t in targets]
            
            # Forward pass: RetinaNet computes classification & regression losses internally
            loss_dict = model(images, targets)
            
            # Sum up the losses (classification loss + bounding box regression loss)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            
        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch [{epoch+1}/{CONFIG['epochs']}] - Total Loss: {avg_loss:.4f} "
                    f"(Cls Loss: {loss_dict['classification'].item():.4f}, "
                    f"Bbox Loss: {loss_dict['bbox_regression'].item():.4f})")

    logger.info("Training Complete!")
    return model

# ==========================================
# 4. EXPORT & PRODUCTION USAGE DEMO
# ==========================================
def production_inference_demo(trained_model):
    logger.info("\n" + "="*50)
    logger.info("DEMO: DETECTING FACES IN A NEW IMAGE")
    logger.info("="*50)
    
    # Switch to eval mode! (Crucial for inference: model now returns predictions instead of losses)
    trained_model.eval()
    
    # Create a dummy image (e.g., a 1080p camera frame)
    dummy_frame = torch.rand(3, 1080, 1920).to(CONFIG["device"])
    
    with torch.no_grad():
        # Model expects a list of images, returns a list of dictionaries
        predictions = trained_model([dummy_frame])
        
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    
    # Filter by confidence threshold (e.g., > 0.5)
    confidence_threshold = 0.5
    valid_detections = scores > confidence_threshold
    
    boxes = boxes[valid_detections]
    scores = scores[valid_detections]
    
    logger.info(f"Found {len(boxes)} faces with confidence > {confidence_threshold}")
    
    for i, (box, score) in enumerate(zip(boxes, scores)):
        logger.info(f"Face {i+1}: BBox {box.astype(int)} | Confidence: {score*100:.1f}%")

    # Save weights for production API deployment
    # torch.save(trained_model.state_dict(), "retinanet_face_detector.pth")

if __name__ == "__main__":
    # 1. Run the fine-tuning process
    trained_detector = train_detector()
    
    # 2. Demonstrate how to run inference for Face Detection
    production_inference_demo(trained_detector)