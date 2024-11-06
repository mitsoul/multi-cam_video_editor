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
    return parser.parse_args()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def create_adjustment_window(angle_idx):
    """Create window with trackbars for brightness and contrast adjustment"""
    window_name = f'Adjust Angle {angle_idx}'
    cv2.namedWindow(window_name)
    # Initialize with neutral values
    cv2.createTrackbar('Brightness', window_name, 0, 200, lambda x: None)  # -100 to +100
    cv2.createTrackbar('Contrast', window_name, 100, 200, lambda x: None)  # 0 to 2.0
    return window_name

def adjust_frame(frame, brightness, contrast):
    """Apply brightness and contrast adjustments to frame"""
    brightness = brightness - 100  # Convert from 0-200 to -100-100
    contrast = contrast / 100.0    # Convert from 0-200 to 0-2.0
    
    adjusted = frame.astype(float)
    adjusted = contrast * adjusted + brightness
    adjusted = np.clip(adjusted, 0, 255)
    return adjusted.astype(np.uint8)

def get_manual_adjustments(sample_frames):
    """Interactive window for manual brightness/contrast adjustment"""
    adjustments = {
        0: {  # Initialize angle 0 with neutral values
            'brightness': 0,
            'contrast': 1.0
        }
    }
    reference_frame = sample_frames[0]  # Use first angle as reference
    cv2.imshow('Reference (Angle 0)', reference_frame)
    
    # Create adjustment windows for other angles
    for i in range(1, len(sample_frames)):
        window_name = create_adjustment_window(i)
        frame = sample_frames[i]
        
        while True:
            brightness = cv2.getTrackbarPos('Brightness', window_name)
            contrast = cv2.getTrackbarPos('Contrast', window_name)
            
            adjusted = adjust_frame(frame, brightness, contrast)
            cv2.imshow(window_name, adjusted)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):  # Press 'n' to move to next angle
                adjustments[i] = {
                    'brightness': brightness - 100,  # Store as -100 to +100
                    'contrast': contrast / 100.0     # Store as 0.0 to 2.0
                }
                break
            elif key == ord('q'):  # Press 'q' to quit
                cv2.destroyAllWindows()
                return None
    
    cv2.destroyAllWindows()
    return adjustments

def calculate_brightness_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    # Calculate the score based on brightness (higher brightness = higher score)
    score = avg_brightness / 255  # Normalize to 0-1 range
    return score

def calculate_target_variables(frames):
    """
    Calculate the target brightness, contrast, hue, saturation, and gamma for a set of frames.
    Returns a dictionary with target values for each video property.
    """
    brightness_list = []
    contrast_list = []
    hue_list = []
    saturation_list = []
    gamma_list = []

    for frame in frames:
        # Convert to LAB and HSV for brightness/contrast and hue/saturation
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Brightness and contrast
        l_channel, a, b = cv2.split(lab)
        brightness = np.mean(l_channel)
        contrast = np.std(l_channel)

        brightness_list.append(brightness)
        contrast_list.append(contrast)

        # Hue and Saturation
        h_channel, s_channel, v_channel = cv2.split(hsv)
        hue = np.mean(h_channel)
        saturation = np.mean(s_channel)

        hue_list.append(hue)
        saturation_list.append(saturation)

        # Gamma estimation (assume luminance correlation for simplicity)
        gamma_list.append(estimate_gamma(frame))

    # Calculate the target as the average for each property
    target_brightness = np.mean(brightness_list)
    target_contrast = np.mean(contrast_list)
    target_hue = np.mean(hue_list)
    target_saturation = np.mean(saturation_list)
    target_gamma = np.mean(gamma_list)

    return {
        'brightness': target_brightness,
        'contrast': target_contrast,
        'hue': target_hue,
        'saturation': target_saturation,
        'gamma': target_gamma
    }

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
    return visibility * 0.5 + centeredness * 0.2 + brightness_score * 0.6 + text_score * 1.2

# Process the videos
def process_videos(video_dir, output_path):
    # Add timestamp tracking
    angle_timestamps = []
    current_time = 0.0

    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]
    video_caps = [cv2.VideoCapture(vf) for vf in video_files]

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
    output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_widths[0], frame_heights[0]))

    # Track progress with tqdm
    total_frames = min(frame_counts)
    frame_duration = 1 / fps  # Duration of each frame in seconds
    hold_duration = 3  # Duration to hold the best angle in seconds
    hold_frames = int(hold_duration / frame_duration)  # Number of frames

    sample_frames = []
    for cap in video_caps:
        ret, frame = cap.read()
        if ret:
            sample_frames.append(frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame

    # Get manual adjustments for each angle
    print("\nAdjust brightness and contrast for each angle.")
    print("Press 'n' to confirm settings for current angle and move to next.")
    print("Press 'q' to quit adjustment process.")
    adjustments = get_manual_adjustments(sample_frames)
    if adjustments is None:
        print("Adjustment process cancelled.")
        return

    with tqdm(total=total_frames, desc="Processing Frames", unit="frame", ncols=100) as pbar:
        # Iterate over frames
        hold_counter = 0
        best_cap_idx = None
        current_frame = None
        last_angle = None
        segment_start = 0.0
        
        for i in range(total_frames):
            current_time = i / fps

            if hold_counter > 0:
                # Continue using the stored best frame index
                angle_usage_count[best_cap_idx] += 1
                for cap_idx, cap in enumerate(video_caps):
                    ret, frame = cap.read()
                    if cap_idx == best_cap_idx and ret:
                        brightness = adjustments[cap_idx]['brightness']
                        contrast = adjustments[cap_idx]['contrast']
                        adjusted_frame = adjust_frame(frame, brightness + 100, int(contrast * 100))
                        output_video.write(adjusted_frame)
                hold_counter -= 1
            else:
                # Reset best score for new comparison
                best_score = -float('inf')
                best_frame = None
                previous_best_cap_idx = best_cap_idx
                
                # Compare all camera angles
                for cap_idx, cap in enumerate(video_caps):
                    ret, frame = cap.read()
                    if not ret:
                        continue

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

                # Write the best frame and reset hold counter
                if best_frame is not None:
                    # If we're switching to a new angle or this is the first frame
                    if best_cap_idx != previous_best_cap_idx or previous_best_cap_idx is None:
                        # Log the previous segment if it exists
                        if previous_best_cap_idx is not None:
                            angle_timestamps.append({
                                'angle': previous_best_cap_idx,
                                'start': format_timestamp(segment_start),
                                'end': format_timestamp(current_time),
                                'video_file': os.path.basename(video_files[previous_best_cap_idx])
                            })
                        segment_start = current_time

                    angle_usage_count[best_cap_idx] += 1
                    brightness = adjustments[best_cap_idx]['brightness']
                    contrast = adjustments[best_cap_idx]['contrast']
                    adjusted_frame = adjust_frame(best_frame, brightness + 100, int(contrast * 100))
                    output_video.write(adjusted_frame)
                    hold_counter = hold_frames - 1

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
    process_videos(args.input_dir, args.output_path)
