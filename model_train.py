import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm

class RoadSegmentationDataset(Dataset):
    """Dataset class for free-space segmentation"""
    def __init__(self, image_paths: List[str], masks: np.ndarray, transform=None):
        self.image_paths = image_paths
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = self.masks[idx]
        
        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(mask).float()
        
        return image, mask

class UNet(nn.Module):
    """U-Net architecture for semantic segmentation"""
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self._make_layer(3, 64)
        self.enc2 = self._make_layer(64, 128)
        self.enc3 = self._make_layer(128, 256)
        self.enc4 = self._make_layer(256, 512)
        
        # Decoder
        self.dec4 = self._make_layer(512 + 256, 256)
        self.dec3 = self._make_layer(256 + 128, 128)
        self.dec2 = self._make_layer(128 + 64, 64)
        self.dec1 = nn.Conv2d(64, 1, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Decoder
        d4 = self.dec4(torch.cat([self.upsample(e4), e3], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e1], dim=1))
        out = torch.sigmoid(self.dec1(d2))
        
        return out

class SegmentationTrainer:
    """Trainer class for segmentation models"""
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
    def train_epoch(self, model, dataloader, criterion, optimizer):
        model.train()
        total_loss = 0
        
        for images, masks in tqdm(dataloader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            optimizer.zero_grad()
            
            if isinstance(model, SegformerForSemanticSegmentation):
                outputs = model(pixel_values=images).logits
            else:
                outputs = model(images)
                
            loss = criterion(outputs, masks.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def validate(self, model, dataloader, criterion):
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, masks in dataloader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                if isinstance(model, SegformerForSemanticSegmentation):
                    outputs = model(pixel_values=images).logits
                else:
                    outputs = model(images)
                    
                loss = criterion(outputs, masks.unsqueeze(1))
                total_loss += loss.item()
                
        return total_loss / len(dataloader)

def visualize_predictions(model, image_path: str, mask: np.ndarray, device: str):
    """Visualize model predictions"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        if isinstance(model, SegformerForSemanticSegmentation):
            pred_mask = model(pixel_values=input_tensor).logits
        else:
            pred_mask = model(input_tensor)
    
    pred_mask = pred_mask.squeeze().cpu().numpy()
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Ground Truth')
    ax3.imshow(pred_mask > 0.5, cmap='gray')
    ax3.set_title('Prediction')
    plt.show()

def main():
    # Dataset parameters
    images = ["test2.jpg", "test23.jpg"]  # Your image paths
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load your generated masks here
    # This should match the masks generated from your previous script
    road_masks = np.load('road_masks.npy')  # Replace with your actual masks
    
    # Create dataset and dataloaders
    dataset = RoadSegmentationDataset(images, road_masks, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)
    
    # Initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # U-Net
    unet = UNet().to(device)
    
    # SegFormer
    segformer_config = SegformerConfig(
        num_channels=3,
        num_encoder_blocks=4,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[32, 64, 160, 256],
        patch_sizes=[7, 3, 3, 3],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        num_labels=1
    )
    segformer = SegformerForSemanticSegmentation(segformer_config).to(device)
    
    # Training parameters
    criterion = nn.BCEWithLogitsLoss()
    unet_optimizer = optim.Adam(unet.parameters(), lr=1e-4)
    segformer_optimizer = optim.Adam(segformer.parameters(), lr=1e-4)
    num_epochs = 50
    
    # Initialize trainer
    trainer = SegmentationTrainer(device)
    
    # Train U-Net
    print("Training U-Net...")
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(unet, train_loader, criterion, unet_optimizer)
        val_loss = trainer.validate(unet, val_loader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Train SegFormer
    print("\nTraining SegFormer...")
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(segformer, train_loader, criterion, segformer_optimizer)
        val_loss = trainer.validate(segformer, val_loader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Visualize results
    for image_path, mask in zip(images, road_masks):
        print(f"\nVisualizing results for {image_path}")
        print("U-Net predictions:")
        visualize_predictions(unet, image_path, mask, device)
        print("SegFormer predictions:")
        visualize_predictions(segformer, image_path, mask, device)

if __name__ == "__main__":
    main()
