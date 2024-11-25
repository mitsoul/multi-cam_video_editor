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


def calculate_target_values(reference_frame):
    """Calculate target brightness and contrast values from a reference frame"""
    # Convert to LAB and calculate CLAHE-enhanced brightness
    lab = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Calculate target values using enhanced L channel
    target_brightness = np.percentile(cl, 90) / 255  # Using 90th percentile
    target_contrast = np.std(cl) / 64  # More aggressive contrast
    target_contrast = np.clip(target_contrast * 2.5, 0.5, 2.0)
    
    return target_brightness, target_contrast

def adjust_frame_to_reference(frame, reference_frame):
    """Adjust frame's brightness and contrast to match reference frame"""
    target_brightness, target_contrast = calculate_target_values(reference_frame)
    
    # Calculate current frame values
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    curr_brightness = np.percentile(cl, 90) / 255
    curr_contrast = np.std(cl) / 64
    curr_contrast = np.clip(curr_contrast * 2.5, 0.5, 2.0)
    
    # Calculate adjustments
    brightness_diff = (target_brightness - curr_brightness) * 1.5
    contrast_ratio = target_contrast / curr_contrast if curr_contrast > 0 else 1.5
    
    # Convert to adjustment parameters
    brightness_adjustment = int(brightness_diff * 250)
    contrast_adjustment = contrast_ratio * 1.2
    
    # Apply adjustments using existing adjust_frame function
    return adjust_frame(frame, 
                       np.clip(brightness_adjustment + 100, 0, 200),
                       int(np.clip(contrast_adjustment * 100, 80, 200)))


def get_manual_adjustments(sample_frames):
    """Interactive window for manual brightness/contrast adjustment"""
    adjustments = {}
    
    # Calculate trackbar-based brightness scores for each frame
    brightness_scores = []
    for frame in sample_frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:,:,2]
        brightness = np.percentile(v_channel, 90) / 255  # Normalize to 0-1
        brightness_trackbar = int(np.clip(brightness * 200, 0, 200))
        brightness_scores.append(brightness_trackbar)
    
    # Find the frame with highest trackbar-based brightness score
    reference_idx = brightness_scores.index(max(brightness_scores))
    reference_frame = sample_frames[reference_idx]
    
    print(f"\nUsing Angle {reference_idx} as reference (brightest frame, score: {brightness_scores[reference_idx]})")
    print("Adjust each angle's brightness and contrast:")
    print("- Press 'n' to confirm settings and move to next angle")
    print("- Press 'q' to cancel adjustment process")
    cv2.imshow(f'Reference (Angle {reference_idx})', reference_frame)
    
    # Initialize reference angle with neutral values
    adjustments[reference_idx] = {
        'brightness': 0,    # Neutral brightness adjustment
        'contrast': 1.0     # Neutral contrast multiplier
    }
    
    # Create adjustment windows for other angles
    for i in range(len(sample_frames)):
        if i == reference_idx:
            continue
            
        window_name = create_adjustment_window(i)
        frame = sample_frames[i]
        
        while True:
            # Get current trackbar positions
            brightness = cv2.getTrackbarPos('Brightness', window_name)
            contrast = cv2.getTrackbarPos('Contrast', window_name)
            
            # Show adjusted frame in real-time
            adjusted = adjust_frame(frame, brightness, contrast)
            cv2.imshow(window_name, adjusted)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):  # Next angle
                adjustments[i] = {
                    'brightness': brightness - 100,  # Store as -100 to +100
                    'contrast': contrast / 100.0     # Store as 0.0 to 2.0
                }
                break
            elif key == ord('q'):  # Quit adjustment
                cv2.destroyAllWindows()
                return None
    
    cv2.destroyAllWindows()
    return adjustments
