"""
SAM (Segment Anything Model) Handler
Interactive segmentation with point/box prompts
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Optional
import os


class SAMHandler:
    """Handler for Segment Anything Model (SAM)"""
    
    def __init__(self, model_type: str = "vit_b", checkpoint_path: str = None):
        """
        Initialize SAM handler
        
        Args:
            model_type: Model variant (vit_b, vit_l, vit_h)
            checkpoint_path: Path to SAM checkpoint
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = None
        self.model_type = model_type
        
        # Load model
        self._load_model(checkpoint_path)
    
    def _load_model(self, checkpoint_path: str = None):
        """Load SAM model from embedded code.
        Resolution order for checkpoint:
        1. Explicit checkpoint_path argument
        2. Environment variable SAM_WEIGHTS
        3. Default local path models/weights/sam_vit_b_01ec64.pth (or matching variant)
        """
        try:
            from .sam.build_sam import sam_model_registry
            from .sam.predictor import SamPredictor

            # Determine filename by variant
            filename_map = {
                "vit_b": "sam_vit_b_01ec64.pth",
                "vit_l": "sam_vit_l_0b3195.pth",
                "vit_h": "sam_vit_h_4b8939.pth",
            }
            default_filename = filename_map.get(self.model_type, filename_map["vit_b"])
            default_path = Path(__file__).parent / "weights" / default_filename

            env_path = os.getenv("SAM_WEIGHTS")
            resolved_checkpoint = Path(checkpoint_path or env_path or default_path)

            if resolved_checkpoint.exists():
                sam = sam_model_registry[self.model_type](checkpoint=str(resolved_checkpoint))
                sam.to(device=self.device)
                self.predictor = SamPredictor(sam)
                print(f"✓ SAM {self.model_type} loaded: {resolved_checkpoint} (device={self.device})")
            else:
                print(f"⚠ SAM checkpoint not found at {resolved_checkpoint}. Interactive segmentation will use fallback.")
        except Exception as e:
            print(f"⚠ Failed to initialize SAM model: {e}. Using fallback segmentation.")
    
    def predict(self,
                image: np.ndarray,
                point_coords: Optional[np.ndarray] = None,
                point_labels: Optional[np.ndarray] = None,
                box: Optional[np.ndarray] = None,
                multimask_output: bool = False) -> np.ndarray:
        """
        Predict segmentation mask with prompts
        
        Args:
            image: Input image (BGR, OpenCV format)
            point_coords: Nx2 array of point coordinates [[x, y], ...]
            point_labels: N array of point labels (1=foreground, 0=background)
            box: Box coordinates [x1, y1, x2, y2]
            multimask_output: Return multiple masks
            
        Returns:
            Binary mask (0-255, uint8)
        """
        if self.predictor is None:
            # Fallback if SAM not loaded
            print("⚠ SAM model not loaded, using fallback method")
            return self._fallback_segmentation(image, point_coords, box)
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image
        self.predictor.set_image(image_rgb)
        
        # Predict with prompts
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask_output
        )
        
        # Get best mask (highest score)
        if multimask_output:
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
        else:
            mask = masks[0]
        
        # Convert to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        return mask_uint8
    
    def _fallback_segmentation(self, 
                               image: np.ndarray,
                               point_coords: Optional[np.ndarray] = None,
                               box: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fallback segmentation using traditional methods
        """
        h, w = image.shape[:2]
        
        # If box provided, use it
        if box is not None:
            mask = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = box.astype(int)
            
            # Use GrabCut with box
            rect = (x1, y1, x2 - x1, y2 - y1)
            temp_mask = np.zeros((h, w), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            cv2.grabCut(image, temp_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            mask = np.where((temp_mask == 2) | (temp_mask == 0), 0, 255).astype('uint8')
            
            return mask
        
        # If points provided, use flood fill
        if point_coords is not None and len(point_coords) > 0:
            mask = np.zeros((h, w), dtype=np.uint8)
            for point in point_coords:
                x, y = int(point[0]), int(point[1])
                # Flood fill from point
                temp_mask = np.zeros((h + 2, w + 2), np.uint8)
                cv2.floodFill(image, temp_mask, (x, y), 255, (10,) * 3, (10,) * 3)
                mask = cv2.bitwise_or(mask, temp_mask[1:-1, 1:-1])
            
            return mask
        
        # Default: return empty mask
        return np.zeros((h, w), dtype=np.uint8)
