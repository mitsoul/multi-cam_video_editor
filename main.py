import cv2
import mediapipe as mp
import numpy as np
import json
from datetime import timedelta
import os
from tqdm import tqdm
import argparse
import aspose.pycore as asposecore
import aspose.imaging as asposeimaging
import pytesseract

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS.mmm format"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    milliseconds = round(seconds % 1 * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{milliseconds:03d}"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process videos to select the best frames based on pose.")
    parser.add_argument('input_dir', type=str, help='Path to the input video directory')
    parser.add_argument('output_path', type=str, help='Path to the output video file')
    parser.add_argument('--cuts_json', type=str, help='Path to JSON file containing predefined cuts (optional)')
    parser.add_argument('--adjustment_mode', type=str, choices=['manual', 'auto', 'none'], 
                       default='auto', help='Choose between manual, automatic, or no brightness/contrast adjustment')
    return parser.parse_args()

def parse_timestamp(timestamp):
    """Convert HH:MM:SS.mmm format to seconds"""
    hours, minutes, seconds = timestamp.split(':')
    seconds, milliseconds = seconds.split('.')
    total_seconds = (
        int(hours) * 3600 + 
        int(minutes) * 60 + 
        int(seconds) + 
        int(milliseconds) / 1000
    )
    return total_seconds

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def create_adjustment_window(angle_idx):
    """Create window with trackbars for brightness and contrast adjustment"""
    window_name = f'Adjust Angle {angle_idx}'
    cv2.namedWindow(window_name)
    # Initialize with neutral values (100 is center/neutral)
    cv2.createTrackbar('Brightness', window_name, 100, 200, lambda x: None)  # 0-200 maps to -100 to +100
    cv2.createTrackbar('Contrast', window_name, 100, 200, lambda x: None)    # 0-200 maps to 0-2.0
    return window_name

def adjust_frame(frame, brightness, contrast):
    """
    Enhanced frame adjustment using CLAHE and traditional brightness/contrast
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge channels
    limg = cv2.merge((cl, a, b))
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Apply additional brightness/contrast adjustments if needed
    alpha = contrast / 100.0  # Contrast (0-2)
    beta = brightness - 100   # Brightness (-100 to +100)
    
    # Apply contrast and brightness using cv2's convertScaleAbs
    final = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
    return final

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

def calculate_brightness_score(frame):
    """Calculate perceptual brightness score using HSV Value channel"""
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Use V channel for brightness
    v_channel = hsv[:, :, 2]
    return np.mean(v_channel) / 255  # Normalize to 0-1 range

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

def estimate_gamma(image):
    """
    Estimate the gamma value of an image based on its luminance.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_luminance = np.mean(gray)

    # Use a heuristic for gamma estimation (assuming average luminance 127 is ideal for gamma 1.0)
    gamma = np.log(mean_luminance / 255) / np.log(127 / 255)
    return gamma


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


def detect_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return len(text.strip())

# Function to calculate visibility score
def visibility_score(landmarks):
    visible = [lm.visibility for lm in landmarks if lm.visibility > 0.5]  # Consider visibility > 0.5
    return len(visible) / len(landmarks)

# Function to calculate centeredness score
def centeredness_score(landmarks, frame_width, frame_height):
    # Assume torso is defined by the midpoint between shoulders and hips
    torso_x = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + 
               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2
    torso_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + 
               landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
    # Calculate distance from the center of the frame
    center_x, center_y = frame_width / 2, frame_height / 2
    distance = np.sqrt((torso_x * frame_width - center_x)**2 + (torso_y * frame_height - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    return 1 - (distance / max_distance)

def has_audio(video_file):
    """Check if a video file has an audio stream using ffmpeg"""
    import subprocess
    
    cmd = ['ffprobe', '-i', video_file, '-show_streams', '-select_streams', 'a', '-loglevel', 'error']
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return bool(result.stdout)

# Function to compute combined score
def compute_score(landmarks, frame_width, frame_height, frame):
    visibility = visibility_score(landmarks)
    centeredness = centeredness_score(landmarks, frame_width, frame_height)
    brightness_score = calculate_brightness_score(frame)
    text_score = detect_text(frame) / 100  # Normalize text score (assuming 100 characters is a good benchmark)
    return visibility * 0.5 + centeredness * 0.2  + text_score * 1.2

# Process the videos
def process_videos(video_dir, output_path, cuts_json=None, adjustment_mode='auto'):
    # Add timestamp tracking
    angle_timestamps = []
    current_time = 0.0

    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]
    video_caps = [cv2.VideoCapture(vf) for vf in video_files]

    # Get FPS for all videos
    fps_values = [cap.get(cv2.CAP_PROP_FPS) for cap in video_caps]
    target_fps = min(fps_values)  # Use the lowest FPS as our target
    
    print("\nFPS Synchronization:")
    print("-" * 50)
    print(f"Target FPS: {target_fps}")
    
    # Calculate frame skip ratios for each video
    frame_skip_ratios = []
    for idx, fps in enumerate(fps_values):
        ratio = round(fps / target_fps)
        frame_skip_ratios.append(ratio)
        print(f"Video {idx}: {fps} fps -> Skip every {ratio} frames")
    
    # Modify the frame reading logic
    def read_synchronized_frames():
        frames = []
        for idx, cap in enumerate(video_caps):
            skip_ratio = frame_skip_ratios[idx]
            ret, frame = cap.read()
            
            # Skip frames for higher FPS videos
            for _ in range(skip_ratio - 1):
                cap.read()  # Read and discard extra frames
                
            if ret:
                frames.append((idx, frame))
        return frames if len(frames) == len(video_caps) else None
    
    # # Add detailed video information printing
    # print("\nVideo Information:")
    # print("-" * 50)
    # for idx, (video_file, cap) in enumerate(zip(video_files, video_caps)):
    #     fps = cap.get(cv2.CAP_PROP_FPS)
    #     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     duration = frame_count / fps if fps > 0 else 0
        
    #     print(f"\nVideo {idx}: {os.path.basename(video_file)}")
    #     print(f"  Frame Rate: {fps:.2f} fps")
    #     print(f"  Frame Count: {frame_count:,}")
    #     print(f"  Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    #     print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        
    #     # Verify frame rate consistency by sampling
    #     timestamps = []
    #     for _ in range(min(100, frame_count)):  # Sample first 100 frames
    #         cap.read()
    #         timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        
    #     if len(timestamps) > 1:
    #         frame_intervals = [j-i for i, j in zip(timestamps[:-1], timestamps[1:])]
    #         avg_interval = sum(frame_intervals) / len(frame_intervals)
    #         measured_fps = 1000 / avg_interval if avg_interval > 0 else 0
    #         print(f"  Measured Frame Rate: {measured_fps:.2f} fps")
            
    #         if abs(measured_fps - fps) > 1:
    #             print(f"  ⚠️ Warning: Reported FPS ({fps:.2f}) differs from measured FPS ({measured_fps:.2f})")
        
    #     # Reset video capture to start
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # print("\n" + "-" * 50)

    # Add angle usage tracking
    angle_usage_count = {i: 0 for i in range(len(video_caps))}

    # Create logs directory if it doesn't exist
    os.makedirs('./logs', exist_ok=True)
    log_file = os.path.join('./logs', f'{os.path.basename(output_path)}_angles.json')
    
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in video_caps]
    frame_widths = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in video_caps]
    frame_heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in video_caps]
    fps = video_caps[0].get(cv2.CAP_PROP_FPS)  # Assuming same FPS for all videos

    # Prepare output video writer
    # Update the output video writer to use the target_fps
    output_video = cv2.VideoWriter(output_path, 
                                 cv2.VideoWriter_fourcc(*'mp4v'), 
                                 target_fps,  # Use synchronized fps
                                 (frame_widths[0], frame_heights[0]))

    # Track progress with tqdm
    total_frames = min(frame_counts)
    frame_duration = 1 / fps  # Duration of each frame in seconds
    hold_duration = 5  # Duration to hold the best angle in seconds
    hold_frames = int(hold_duration / frame_duration)  # Number of frames

    sample_frames = []
    for cap in video_caps:
        ret, frame = cap.read()
        if ret:
            sample_frames.append(frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame

    # Find reference angle (brightest frame) using existing method
    reference_angle = None
    brightness_scores = []
    for frame in sample_frames:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        brightness = np.percentile(cl, 90) / 255
        brightness_trackbar = int(np.clip(brightness * 200, 0, 200))
        brightness_scores.append(brightness_trackbar)
    
    reference_angle = brightness_scores.index(max(brightness_scores))
    print(f"\nUsing Angle {reference_angle} as reference for dynamic adjustments")

    # Get manual adjustments for each angle
    if adjustment_mode == 'manual':
        adjustments = get_manual_adjustments(sample_frames)
        if adjustments is None:
            print("Adjustment process cancelled.")
            return
    elif adjustment_mode == 'auto':
        # Calculate automatic adjustments using angle 0 as reference
        adjustments = calculate_target_variables(sample_frames)
        print("\nAutomatic adjustments calculated relative to Angle 0")
    else:  # adjustment_mode == 'none'
        # Use neutral values for no adjustment
        adjustments = {
            i: {
                'brightness': 0,    # Neutral brightness
                'contrast': 1.0     # Neutral contrast
            }
            for i in range(len(sample_frames))
        }
        print("No brightness/contrast adjustments will be applied")
    
    if cuts_json:
        print("yes")
        with open(cuts_json, 'r') as f:
            cuts_data = json.load(f)
            angle_segments = cuts_data.get('angle_segments', [])
    
        with tqdm(total=total_frames, desc="Processing Frames", unit="frame", ncols=100) as pbar:
            frame_idx = 0
            current_segment_idx = 0
            
            while frame_idx < total_frames and current_segment_idx < len(angle_segments):
                segment = angle_segments[current_segment_idx]
                current_time = frame_idx / target_fps  # Use target_fps here
                segment_start = parse_timestamp(segment['start'])
                segment_end = parse_timestamp(segment['end'])
                angle_idx = segment['angle']

                if current_time >= segment_start and current_time < segment_end:
                    frames = read_synchronized_frames()
                    if frames:
                        reference_frame = None
                        for cap_idx, frame in frames:
                            if cap_idx == reference_angle:  # Use determined reference angle
                                reference_frame = frame
                                break
                                
                        for cap_idx, frame in frames:
                            if cap_idx == angle_idx:
                                adjusted_frame = adjust_frame_to_reference(frame, reference_frame)
                                output_video.write(adjusted_frame)
                                angle_usage_count[angle_idx] += 1
                    frame_idx += 1
                    pbar.update(1)
                else:
                    current_segment_idx += 1
    
    else:
        print("no")
        with tqdm(total=total_frames, desc="Processing Frames", unit="frame", ncols=100) as pbar:
            hold_counter = 0
            best_cap_idx = None
            segment_start = 0.0
            
            for i in range(total_frames):
                current_time = i / target_fps

                if hold_counter > 0:
                    frames = read_synchronized_frames()
                    if frames:
                        reference_frame = None
                        for cap_idx, frame in frames:
                            if cap_idx == reference_angle:  # Use determined reference angle
                                reference_frame = frame
                                break
                                
                        for cap_idx, frame in frames:
                            if cap_idx == angle_idx:
                                adjusted_frame = adjust_frame_to_reference(frame, reference_frame)
                                output_video.write(adjusted_frame)
                                angle_usage_count[angle_idx] += 1
                    hold_counter -= 1
                else:
                    best_score = -float('inf')
                    best_frame = None
                    frames = read_synchronized_frames()
                    
                    if frames:
                        for cap_idx, frame in frames:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = pose.process(frame_rgb)
                            score = 0

                            if results.pose_landmarks:
                                landmarks = results.pose_landmarks.landmark
                                score = compute_score(landmarks, frame_widths[cap_idx], frame_heights[cap_idx], frame)

                            if score > best_score:
                                best_score = score
                                best_cap_idx = cap_idx
                                best_frame = frame

                        reference_frame = None
                        for cap_idx, frame in frames:
                            if cap_idx == reference_angle:  # Use determined reference angle
                                reference_frame = frame
                                break
                                
                        if best_frame is not None:
                            adjusted_frame = adjust_frame_to_reference(best_frame, reference_frame)
                            output_video.write(adjusted_frame)
                            angle_usage_count[best_cap_idx] += 1

                pbar.update(1)
            
            # Log the final segment
            if best_cap_idx is not None:
                angle_timestamps.append({
                    'angle': best_cap_idx,
                    'start': format_timestamp(segment_start),
                    'end': format_timestamp(current_time),
                    'video_file': os.path.basename(video_files[best_cap_idx])
                })

    # Find the most used angle that has audio
    most_used_angle = None
    max_usage = -1
    for angle_idx, count in angle_usage_count.items():
        if count > max_usage and has_audio(video_files[angle_idx]):
            max_usage = count
            most_used_angle = angle_idx

    print("\nAngle usage statistics:")
    for angle_idx, count in angle_usage_count.items():
        percentage = (count / total_frames) * 100
        has_audio_str = " (has audio)" if has_audio(video_files[angle_idx]) else " (no audio)"
        print(f"Angle {angle_idx}: {count} frames ({percentage:.1f}%){has_audio_str}")
    print(f"\nSelected audio from angle {most_used_angle}")

    # Release all resources before file operations
    output_video.release()
    for cap in video_caps:
        cap.release()

    # Extract and combine video with audio using ffmpeg
    temp_output = output_path.replace('.mp4', '_temp.mp4')
    
    # Make sure the file handles are released before renaming
    import time
    time.sleep(1)  # Give the system time to release file handles
    
    os.rename(output_path, temp_output)
    ffmpeg_cmd = f'ffmpeg -i "{temp_output}" -i "{video_files[most_used_angle]}" -c:v copy -map 0:v:0 -map 1:a:0 "{output_path}"'
    os.system(ffmpeg_cmd)
    os.remove(temp_output)

    # Release all resources
    output_video.release()
    for cap in video_caps:
        cap.release()

    # Save the angle timestamps to JSON
    log_data = {
        'output_video': os.path.basename(output_path),
        'fps': fps,
        'total_frames': total_frames,
        'angle_segments': angle_timestamps,
        'angle_statistics': {
            str(angle): {
                'frame_count': count,
                'percentage': (count / total_frames) * 100,
                'has_audio': has_audio(video_files[angle])
            }
            for angle, count in angle_usage_count.items()
        }
    }
    
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)

if __name__ == "__main__":
    args = parse_arguments()
    process_videos(args.input_dir, args.output_path, args.cuts_json, args.adjustment_mode)