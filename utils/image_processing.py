"""
Image Processing Utilities
Photo editing adjustments and filters
"""

import cv2
import numpy as np
from typing import Tuple


def adjust_brightness_contrast(img: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    """
    Adjust brightness and contrast
    
    Args:
        img: Input image (BGR)
        brightness: -100 to 100
        contrast: -100 to 100
    """
    # Normalize inputs
    alpha = 1.0 + (contrast / 100.0)  # Contrast control (1.0-3.0)
    beta = brightness * 2.55  # Brightness control (0-100)
    
    # Apply adjustment
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted


def adjust_exposure(img: np.ndarray, exposure: float) -> np.ndarray:
    """
    Adjust exposure (-2 to +2 stops)
    
    Args:
        img: Input image (BGR)
        exposure: -2 to 2 (EV stops)
    """
    # Convert to float
    img_float = img.astype(np.float32) / 255.0
    
    # Apply exposure adjustment (2^exposure)
    adjusted = img_float * (2.0 ** exposure)
    
    # Clip and convert back
    adjusted = np.clip(adjusted * 255, 0, 255).astype(np.uint8)
    return adjusted


def adjust_saturation(img: np.ndarray, saturation: float) -> np.ndarray:
    """
    Adjust color saturation
    
    Args:
        img: Input image (BGR)
        saturation: -100 to 100
    """
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Adjust saturation channel
    scale = 1.0 + (saturation / 100.0)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
    
    # Convert back to BGR
    adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return adjusted


def adjust_temperature_tint(img: np.ndarray, temperature: float, tint: float) -> np.ndarray:
    """
    Adjust color temperature and tint
    
    Args:
        img: Input image (BGR)
        temperature: -100 to 100 (cool to warm)
        tint: -100 to 100 (green to magenta)
    """
    img_float = img.astype(np.float32)
    
    # Temperature adjustment (blue-yellow axis)
    temp_factor = temperature / 100.0
    img_float[:, :, 0] -= temp_factor * 50  # Blue channel
    img_float[:, :, 2] += temp_factor * 50  # Red channel
    
    # Tint adjustment (green-magenta axis)
    tint_factor = tint / 100.0
    img_float[:, :, 1] += tint_factor * 50  # Green channel
    
    # Clip and convert back
    adjusted = np.clip(img_float, 0, 255).astype(np.uint8)
    return adjusted


def adjust_highlights_shadows(img: np.ndarray, highlights: float, shadows: float) -> np.ndarray:
    """
    Adjust highlights and shadows separately
    
    Args:
        img: Input image (BGR)
        highlights: -100 to 100
        shadows: -100 to 100
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]
    
    # Create masks for highlights and shadows
    highlight_mask = (l_channel > 127).astype(np.float32)
    shadow_mask = (l_channel <= 127).astype(np.float32)
    
    # Smooth masks
    highlight_mask = cv2.GaussianBlur(highlight_mask, (21, 21), 0)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)
    
    # Apply adjustments
    highlight_adjust = highlights * 0.5
    shadow_adjust = shadows * 0.5
    
    l_channel += highlight_mask * highlight_adjust
    l_channel += shadow_mask * shadow_adjust
    
    # Clip and update
    lab[:, :, 0] = np.clip(l_channel, 0, 255)
    
    # Convert back to BGR
    adjusted = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return adjusted


def apply_vignette(img: np.ndarray, strength: float) -> np.ndarray:
    """
    Apply vignette effect
    
    Args:
        img: Input image (BGR)
        strength: 0 to 100
    """
    h, w = img.shape[:2]
    
    # Create radial gradient mask
    X_resultant_kernel = cv2.getGaussianKernel(w, w / 2)
    Y_resultant_kernel = cv2.getGaussianKernel(h, h / 2)
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / kernel.max()
    
    # Adjust strength
    strength_factor = strength / 100.0
    mask = mask * (1 - strength_factor) + strength_factor
    
    # Apply mask to each channel
    vignette = img.copy().astype(np.float32)
    for i in range(3):
        vignette[:, :, i] *= mask
    
    return np.clip(vignette, 0, 255).astype(np.uint8)


def composite_with_alpha(img: np.ndarray, mask: np.ndarray, background_color: Tuple[int, int, int] = None) -> np.ndarray:
    """
    Create RGBA image with alpha channel from mask
    
    Args:
        img: Input image (BGR)
        mask: Alpha mask (0-255, grayscale)
        background_color: Optional background color (B, G, R)
    """
    # Ensure mask is single channel
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Create RGBA image
    b, g, r = cv2.split(img)
    rgba = cv2.merge([b, g, r, mask])
    
    return rgba


def refine_mask_morphology(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Refine mask using morphological operations
    
    Args:
        mask: Binary mask
        kernel_size: Kernel size for morphology
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Remove noise with opening
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Fill holes with closing
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return mask


def feather_mask(mask: np.ndarray, feather_amount: int = 5) -> np.ndarray:
    """
    Feather (blur) mask edges
    
    Args:
        mask: Binary mask
        feather_amount: Blur kernel size
    """
    if feather_amount > 0:
        # Make kernel size odd
        kernel_size = feather_amount * 2 + 1
        feathered = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        return feathered
    return mask


def apply_filter_preset(img: np.ndarray, preset: str) -> np.ndarray:
    """
    Apply preset filter to image
    
    Args:
        img: Input image (BGR)
        preset: Filter name
    """
    result = img.copy()
    
    if preset == "vintage":
        # Warm tones, faded look
        result = adjust_temperature_tint(result, 30, 10)
        result = adjust_saturation(result, -20)
        result = adjust_brightness_contrast(result, 10, -15)
        result = apply_vignette(result, 30)
        
    elif preset == "cinematic":
        # Teal & orange, high contrast
        result = adjust_temperature_tint(result, 15, -10)
        result = adjust_saturation(result, 20)
        result = adjust_brightness_contrast(result, 0, 25)
        result = apply_vignette(result, 20)
        
    elif preset == "bw":
        # Black & white
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        result = adjust_brightness_contrast(result, 5, 20)
        
    elif preset == "warm":
        # Warm tones
        result = adjust_temperature_tint(result, 40, 5)
        result = adjust_saturation(result, 15)
        
    elif preset == "cool":
        # Cool tones
        result = adjust_temperature_tint(result, -40, -5)
        result = adjust_saturation(result, 10)
        
    elif preset == "dramatic":
        # High contrast, deep shadows
        result = adjust_brightness_contrast(result, -10, 40)
        result = adjust_highlights_shadows(result, -20, -30)
        result = apply_vignette(result, 40)
        
    elif preset == "soft":
        # Low contrast, pastel
        result = adjust_brightness_contrast(result, 15, -20)
        result = adjust_saturation(result, -15)
        
    elif preset == "vivid":
        # Boosted saturation
        result = adjust_saturation(result, 40)
        result = adjust_brightness_contrast(result, 5, 15)
        
    elif preset == "sepia":
        # Sepia tone
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # Apply sepia tone
        result[:, :, 0] = np.clip(result[:, :, 0] * 0.6, 0, 255)  # Blue
        result[:, :, 1] = np.clip(result[:, :, 1] * 0.8, 0, 255)  # Green
        result[:, :, 2] = np.clip(result[:, :, 2] * 1.0, 0, 255)  # Red
        
    elif preset == "fade":
        # Washed out, retro
        result = adjust_brightness_contrast(result, 20, -30)
        result = adjust_saturation(result, -25)
        result = apply_vignette(result, 25)
    
    return result
