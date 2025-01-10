import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score, f1_score

def load_image(image_path: str, size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """
    Load and preprocess an image for model input.
    
    Args:
        image_path: Path to the image file
        size: Desired size of the output image
    
    Returns:
        Preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image)

def calculate_metrics(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> dict:
    """
    Calculate segmentation metrics (precision, recall, F1).
    
    Args:
        pred_mask: Predicted mask
        true_mask: Ground truth mask
    
    Returns:
        Dictionary containing the metrics
    """
    pred_mask = (pred_mask > 0.5).float().cpu().numpy().flatten()
    true_mask = true_mask.cpu().numpy().flatten()
    
    return {
        'precision': precision_score(true_mask, pred_mask, zero_division=0),
        'recall': recall_score(true_mask, pred_mask, zero_division=0),
        'f1': f1_score(true_mask, pred_mask, zero_division=0)
    }

def save_model_checkpoint(model: torch.nn.Module, 
                         optimizer: torch.optim.Optimizer,
                         epoch: int,
                         loss: float,
                         path: str) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer state to save
        epoch: Current epoch number
        loss: Current loss value
        path: Path to save the checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_model_checkpoint(model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         path: str) -> Tuple[torch.nn.Module, torch.optim.Optimizer, int, float]:
    """
    Load model checkpoint.
    
    Args:
        model: The model to load weights into
        optimizer: The optimizer to load state into
        path: Path to the checkpoint file
    
    Returns:
        Tuple of (model, optimizer, epoch, loss)
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch'], checkpoint['loss']

def plot_training_history(train_losses: List[float], 
                         val_losses: List[float],
                         metrics: Optional[dict] = None) -> None:
    """
    Plot training history and metrics.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        metrics: Optional dictionary of additional metrics to plot
    """
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    
    # Plot metrics if provided
    if metrics:
        plt.subplot(1, 2, 2)
        for metric_name, metric_values in metrics.items():
            plt.plot(metric_values, label=metric_name)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.title('Metrics History')
    
    plt.tight_layout()
    plt.show()

def create_mask_overlay(image: Union[str, Image.Image],
                       mask: np.ndarray,
                       alpha: float = 0.5) -> Image.Image:
    """
    Create an overlay of the segmentation mask on the original image.
    
    Args:
        image: Path to image or PIL Image object
        mask: Binary segmentation mask
        alpha: Transparency of the overlay
    
    Returns:
        PIL Image with mask overlay
    """
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    
    mask_overlay = Image.fromarray((mask * 255).astype(np.uint8))
    mask_overlay = mask_overlay.convert('RGB')
    mask_overlay = Image.blend(image, mask_overlay, alpha)
    
    return mask_overlay

def batch_predict(model: torch.nn.Module,
                 image_paths: List[str],
                 device: str = 'cuda') -> List[np.ndarray]:
    """
    Perform batch prediction on multiple images.
    
    Args:
        model: Trained model
        image_paths: List of paths to images
        device: Device to run predictions on
    
    Returns:
        List of predicted masks
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for image_path in image_paths:
            image = load_image(image_path).unsqueeze(0).to(device)
            pred = model(image).squeeze().cpu().numpy()
            predictions.append(pred > 0.5)
    
    return predictions
