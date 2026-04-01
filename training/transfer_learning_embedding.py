import os
import logging
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

# In production, install via: pip install facenet-pytorch torchvision
try:
    from facenet_pytorch import InceptionResNetV1
except ImportError:
    raise ImportError("Please install facenet_pytorch: pip install facenet-pytorch")

# ==========================================
# 0. CONFIGURATION & LOGGING
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger("TransferLearning")

CONFIG = {
    "data_dir": "./mock_face_dataset",
    "batch_size": 16,
    "epochs": 5,
    "learning_rate": 1e-4,
    "embedding_dim": 512,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================
class FineTunedFaceNet(nn.Module):
    """
    Wraps the standard FaceNet architecture for Transfer Learning.
    Uses a Classification Head for training, but can bypass it to return 512D embeddings.
    """
    def __init__(self, num_classes: int, freeze_base: bool = True):
        super(FineTunedFaceNet, self).__init__()
        
        logger.info("Loading pre-trained InceptionResNetV1 (VGGFace2)...")
        # Base FaceNet outputs a 512-dimensional embedding
        self.backbone = InceptionResNetV1(pretrained='vggface2', classify=False)
        
        if freeze_base:
            logger.info("Freezing early layers. Unfreezing top blocks for domain adaptation.")
            # Freeze all parameters first
            for param in self.backbone.parameters():
                param.requires_grad = False
                
            # Unfreeze the last convolutional block (Block8) and the embedding layer
            # This allows the network to adapt high-level features to your specific dataset
            for param in self.backbone.block8.parameters():
                param.requires_grad = True
            for param in self.backbone.last_linear.parameters():
                param.requires_grad = True
            for param in self.backbone.last_bn.parameters():
                param.requires_grad = True

        # Temporary Classification Head used ONLY during training
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(CONFIG["embedding_dim"], num_classes)
        )

    def forward(self, x, extract_embedding=False):
        """
        If extract_embedding=True, bypasses the classifier and returns the 512D vector.
        Otherwise, returns logits for CrossEntropy training.
        """
        # Get the 512D embedding from the backbone
        embeddings = self.backbone(x)
        
        if extract_embedding:
            # L2 Normalize the embeddings (Crucial for Cosine Similarity in production)
            return torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
        # Pass through classification head for training
        logits = self.classifier(embeddings)
        return logits

# ==========================================
# 2. DATASET PREPARATION & MOCKING
# ==========================================
def create_mock_dataset(base_dir: str, num_classes: int = 5, imgs_per_class: int = 10):
    """Creates dummy images so the script is completely self-contained and runnable."""
    if os.path.exists(base_dir):
        return
        
    logger.info(f"Creating mock dataset at {base_dir} with {num_classes} classes...")
    os.makedirs(base_dir, exist_ok=True)
    for i in range(num_classes):
        class_dir = os.path.join(base_dir, f"person_{i:02d}")
        os.makedirs(class_dir, exist_ok=True)
        for j in range(imgs_per_class):
            # Create a random RGB image mimicking a 160x160 face crop
            img_array = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(os.path.join(class_dir, f"img_{j:03d}.jpg"))

def get_dataloaders(data_dir: str):
    """Sets up PyTorch DataLoaders with FaceNet standard transformations."""
    # FaceNet standard input: 160x160 pixels, normalized.
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(p=0.5), # Basic augmentation
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # facenet-pytorch standard
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    num_classes = len(dataset.classes)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=2,
        drop_last=True
    )
    
    return dataloader, num_classes, dataset.classes

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_model():
    logger.info(f"Using device: {CONFIG['device']}")
    
    # 1. Prep Data
    create_mock_dataset(CONFIG["data_dir"])
    dataloader, num_classes, class_names = get_dataloaders(CONFIG["data_dir"])
    logger.info(f"Found {num_classes} classes: {class_names}")

    # 2. Init Model, Loss, Optimizer
    model = FineTunedFaceNet(num_classes=num_classes, freeze_base=True)
    model = model.to(CONFIG["device"])
    
    criterion = nn.CrossEntropyLoss()
    # Only optimize parameters that require gradients (the unfreezed layers & classifier)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG["learning_rate"])

    # 3. Training Loop
    logger.info("Starting Transfer Learning...")
    for epoch in range(CONFIG["epochs"]):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(CONFIG["device"])
            labels = labels.to(CONFIG["device"])
            
            optimizer.zero_grad()
            
            # Forward pass (classification mode)
            outputs = model(inputs, extract_embedding=False)
            loss = criterion(outputs, labels)
            
            # Backward pass & step
            loss.backward()
            optimizer.step()
            
            # Metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        logger.info(f"Epoch [{epoch+1}/{CONFIG['epochs']}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")

    logger.info("Training Complete!")
    return model

# ==========================================
# 4. EXPORT & PRODUCTION USAGE DEMO
# ==========================================
def production_inference_demo(trained_model):
    logger.info("\n" + "="*50)
    logger.info("DEMO: EXTRACTING EMBEDDINGS FOR VECTOR DB")
    logger.info("="*50)
    
    trained_model.eval() # Set to evaluation mode (disables dropout, fixes batchnorm)
    
    # Create a dummy tensor representing a single cropped face image (1, Channels, Height, Width)
    dummy_face_crop = torch.randn(1, 3, 160, 160).to(CONFIG["device"])
    
    with torch.no_grad():
        # Notice we set extract_embedding=True. 
        # This completely ignores the Softmax head we just trained.
        embedding = trained_model(dummy_face_crop, extract_embedding=True)
        
    embedding_np = embedding.cpu().numpy()[0]
    
    logger.info(f"Extracted Embedding Shape: {embedding_np.shape}")
    logger.info(f"Vector L2 Norm (Should be ~1.0): {np.linalg.norm(embedding_np):.4f}")
    logger.info(f"Sample values (first 5 dims): {embedding_np[:5]}")
    logger.info("This 512D vector is now ready to be inserted into FAISS or Milvus.")
    
    # In a real workflow, you would save the backbone weights:
    # torch.save(trained_model.backbone.state_dict(), "finetuned_facenet_backbone.pth")

if __name__ == "__main__":
    # 1. Run the fine-tuning process
    finetuned_model = train_model()
    
    # 2. Demonstrate how to use the fine-tuned model in the Orchestrator
    production_inference_demo(finetuned_model)