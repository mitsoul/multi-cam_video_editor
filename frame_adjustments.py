import cv2
import numpy as np
import aspose.imaging as asposeimaging

def create_adjustment_window(angle_idx):
    """Create window with trackbars for brightness and contrast adjustment"""
    window_name = f'Adjust Angle {angle_idx}'
    cv2.namedWindow(window_name)
    cv2.createTrackbar('Brightness', window_name, 100, 200, lambda x: None)
    cv2.createTrackbar('Contrast', window_name, 100, 200, lambda x: None)
    return window_name

def adjust_frame(frame, brightness, contrast):
    """Enhanced frame adjustment using CLAHE and traditional brightness/contrast"""
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge and convert back
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Apply additional adjustments
    alpha = contrast / 100.0
    beta = brightness - 100
    final = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
    return final

def calculate_brightness_score(frame):
    """Calculate perceptual brightness score using HSV Value channel"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    return np.mean(v_channel) / 255

def calculate_target_variables(frames):
    """Calculate target brightness and contrast using CLAHE-enhanced reference frame"""
    # Find the brightest frame to use as reference
    brightness_scores = []
    for frame in frames:
        # Apply CLAHE enhancement before brightness calculation
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        brightness = np.percentile(cl, 90) / 255  # Normalize to 0-1
        brightness_trackbar = int(np.clip(brightness * 200, 0, 200))
        brightness_scores.append(brightness_trackbar)
    
    # Find the frame with highest trackbar-based brightness score
    reference_idx = brightness_scores.index(max(brightness_scores))
    reference_frame = frames[reference_idx]
    
    reference_hsv = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2HSV)
    reference_v = reference_hsv[:,:,2]
    
    # Modify reference value calculations
    ref_brightness = np.percentile(reference_v, 90) / 255  # Using 90th percentile
    # Increase contrast sensitivity and base multiplier
    ref_contrast = np.std(reference_v) / 64  # Changed from 128 to 64 for more aggressive contrast
    ref_contrast = np.clip(ref_contrast * 2.5, 0.5, 2.0)  # Increased multiplier from 1.5 

    # Convert to trackbar scale
    ref_brightness_trackbar = int(np.clip(ref_brightness * 200, 0, 200))
    ref_contrast_trackbar = int(np.clip(ref_contrast * 200, 80, 200))
    
    print("\nAnalyzing frame characteristics:")
    print(f"Reference (Angle {reference_idx} - brightest frame):")
    print(f"  Brightness: {ref_brightness_trackbar} (trackbar units), {ref_brightness:.3f} (normalized)")
    print(f"  Contrast: {ref_contrast_trackbar} (trackbar units), {ref_contrast:.3f} (normalized)")
    
    adjustments = {}
    # Set reference angle to neutral values
    adjustments[reference_idx] = {
        'brightness': 0,    # Neutral brightness adjustment
        'contrast': 1.0     # Neutral contrast multiplier
    }
    
    # Calculate adjustments for other angles relative to reference
    for i in range(len(frames)):
        if i == reference_idx:
            continue
            
        frame = frames[i]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:,:,2]
        
        # Modify current frame calculations to match reference calculations
        curr_brightness = np.percentile(v_channel, 90) / 255
        curr_contrast = np.std(v_channel) / 64  # Match reference calculation
        curr_contrast = np.clip(curr_contrast * 2.5, 0.5, 2.0)  # Match reference multiplier
        
        # Make brightness adjustment more aggressive
        brightness_diff = (ref_brightness - curr_brightness) * 1.5  # Added multiplier
        contrast_ratio = ref_contrast / curr_contrast if curr_contrast > 0 else 1.5
        
        # Convert to adjustment values with more aggressive scaling
        brightness_adjustment = int(brightness_diff * 250)  # Increased from 200 to 250
        contrast_adjustment = contrast_ratio * 1.2  # Added multiplier
        
        # Store adjustments with wider ranges
        adjustments[i] = {
            'brightness': np.clip(brightness_adjustment, -100, 100),
            'contrast': np.clip(contrast_adjustment, 0.8, 2.5)  # Increased upper limit from 2.0 to 2.5
        }
        
        # Print analysis for this angle
        print(f"\nAngle {i}:")
        print(f"  Original Brightness: {int(curr_brightness * 200)} (trackbar units), {curr_brightness:.3f} (normalized)")
        print(f"  Original Contrast: {int(curr_contrast * 200)} (trackbar units), {curr_contrast:.3f} (normalized)")
        print(f"  Adjustment - Brightness: {brightness_adjustment:+d}, Contrast: {contrast_adjustment:.2f}x")

    return adjustments
    
def adjust_brightness_contrast(image, target_brightness, target_contrast):
    # Convert the OpenCV image to a format compatible with Aspose.Imaging
    image_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    aspose_image = asposeimaging.Image.from_array(image_data)
    
    # Convert to raster image for adjustment operations
    raster_image = aspose_image.to_raster_image()
    
    # Adjust brightness and contrast separately
    # Scale the target values appropriately (assuming they're in 0-255 range)
    brightness_adjustment = int(target_brightness - 127)  # Center around middle value
    contrast_adjustment = int(target_contrast - 127)      # Center around middle value
    
    raster_image.adjust_brightness(brightness_adjustment)
    raster_image.adjust_contrast(contrast_adjustment)
    
    # Convert back to OpenCV format
    adjusted_image_data = raster_image.to_array()
    adjusted_image = cv2.cvtColor(adjusted_image_data, cv2.COLOR_RGB2BGR)
    
    return adjusted_image