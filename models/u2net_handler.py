"""
U2Net Handler for Automatic Background Removal
Lightweight salient object detection - CPU friendly
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from pathlib import Path
import os

# Local embedded U2Net implementation (copied into models/u2net)
# We no longer rely on an external cloned repository.


class U2NetHandler:
    """Handler for U2Net salient object detection model"""
    
    def __init__(self, model_path: str = None, use_light: bool = True):
        """
        Initialize U2Net handler
        
        Args:
            model_path: Path to model weights (optional, will download if None)
            use_light: Use U2NetP (lighter version) for faster CPU inference
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.use_light = use_light
        
        # Image preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load model
        self._load_model(model_path)
    
    def _load_model(self, model_path: str = None):
        """Load U2Net / U2NetP model weights from embedded code.
        Order of resolution:
        1. Explicit model_path argument
        2. Environment variable U2NET_WEIGHTS
        3. Default local path models/weights/(u2netp|u2net).pth
        """
        try:
            # Import classes from local embedded source
            from .u2net.u2net import U2NET, U2NETP

            # Determine default filename based on variant
            filename = "u2netp.pth" if self.use_light else "u2net.pth"
            default_path = Path(__file__).parent / "weights" / filename

            # Resolve final weight path
            env_path = os.getenv("U2NET_WEIGHTS")
            weights_path = Path(model_path or env_path or default_path)

            # Instantiate model architecture
            self.model = (U2NETP(3, 1) if self.use_light else U2NET(3, 1))

            if weights_path.exists():
                self.model.load_state_dict(torch.load(str(weights_path), map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                print(f"✓ U2Net{'P' if self.use_light else ''} loaded: {weights_path} (device={self.device})")
            else:
                print(f"⚠ U2Net weights not found at {weights_path}. Using fallback segmentation.")
        except Exception as e:
            print(f"⚠ Failed to initialize U2Net model: {e}. Using fallback segmentation.")
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict salient object mask
        
        Args:
            image: Input image (BGR, OpenCV format)
            
        Returns:
            Binary mask (0-255, uint8)
        """
        if self.model is None:
            # Fallback: simple thresholding if model not loaded
            print("⚠ U2Net model not loaded, using fallback method")
            return self._fallback_segmentation(image)
        
        original_h, original_w = image.shape[:2]
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            d1, d2, d3, d4, d5, d6, d7 = self.model(input_tensor)
            pred = d1[:, 0, :, :]  # Take first output
            pred = F.interpolate(pred.unsqueeze(1), 
                                size=(original_h, original_w),
                                mode='bilinear', 
                                align_corners=False)
            pred = pred.squeeze().cpu().numpy()
        
        # Normalize to 0-255
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        mask = (pred * 255).astype(np.uint8)
        
        return mask
    
    def _fallback_segmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Simple fallback segmentation if model not available
        Uses GrabCut for basic background removal
        """
        # Create initial mask
        mask = np.zeros(image.shape[:2], np.uint8)
        
        # Define rectangle (rough center area)
        h, w = image.shape[:2]
        rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))
        
        # GrabCut temporary arrays
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Apply GrabCut
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # Create binary mask
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        result_mask = mask2 * 255
        
        return result_mask
