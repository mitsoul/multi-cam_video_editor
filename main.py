import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm
import argparse
import pytesseract
from frame_processing import correction

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process videos to select the best frames based on pose.")
    parser.add_argument('input_dir', type=str, help='Path to the input video directory')
    parser.add_argument('output_path', type=str, help='Path to the output video file')
    return parser.parse_args()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

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


def adjust_brightness_contrast(image, target_brightness, target_contrast, target_hue=0, target_saturation=1, gamma=1.0):
    # Convert to LAB to adjust brightness and contrast
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Adjust brightness
    if target_brightness - np.mean(l_channel) < 0:
        l_channel = cv2.add(l_channel, target_brightness - np.mean(l_channel))

    # Adjust contrast
    l_channel = cv2.multiply(l_channel, target_contrast / np.std(l_channel))

    # Merge LAB channels back
    lab_adjusted = cv2.merge((l_channel, a, b))

    return cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)


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
def compute_score(landmarks, frame_width, frame_height, frame, target_brightness):
    visibility = visibility_score(landmarks)
    centeredness = centeredness_score(landmarks, frame_width, frame_height)
    brightness_score = calculate_brightness_score(frame)
    text_score = detect_text(frame) / 100  # Normalize text score (assuming 100 characters is a good benchmark)
    return visibility * 0.5 + centeredness * 0.2 + brightness_score * 0.6 + text_score * 1.2

# Process the videos
def process_videos(video_dir, output_path):
    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]
    video_caps = [cv2.VideoCapture(vf) for vf in video_files]

    # Add angle usage tracking
    angle_usage_count = {i: 0 for i in range(len(video_caps))}
    
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

    # Calculate target variables based on the sample frames
    target_variables = calculate_target_variables(sample_frames)

    # Use these target variables in your adjust_brightness_contrast function
    target_brightness = target_variables['brightness']
    target_contrast = target_variables['contrast']
    target_hue = target_variables['hue']
    target_saturation = target_variables['saturation']
    target_gamma = target_variables['gamma']
 

    with tqdm(total=total_frames, desc="Processing Frames", unit="frame", ncols=100) as pbar:
        # Iterate over frames
        hold_counter = 0
        best_cap_idx = None
        current_frame = None
        
        for i in range(total_frames):
            if hold_counter > 0:
                # Continue using the stored best frame index
                angle_usage_count[best_cap_idx] += 1
                for cap_idx, cap in enumerate(video_caps):
                    ret, frame = cap.read()
                    if cap_idx == best_cap_idx and ret:
                        adjusted_frame = adjust_brightness_contrast(frame, target_brightness, target_contrast, target_hue, target_saturation, target_gamma)
                        output_video.write(adjusted_frame)
                hold_counter -= 1
            else:
                # Reset best score for new comparison
                best_score = -float('inf')
                best_frame = None
                
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
                        score = compute_score(landmarks, frame_widths[cap_idx], frame_heights[cap_idx], frame, target_brightness)

                    if score > best_score:
                        best_score = score
                        best_cap_idx = cap_idx
                        best_frame = frame

                # Write the best frame and reset hold counter
                if best_frame is not None:
                    angle_usage_count[best_cap_idx] += 1
                    adjusted_frame = adjust_brightness_contrast(best_frame, target_brightness, target_contrast, target_hue, target_saturation, target_gamma)
                    output_video.write(adjusted_frame)
                    hold_counter = hold_frames - 1

            pbar.update(1)

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

if __name__ == "__main__":
    args = parse_arguments()
    process_videos(args.input_dir, args.output_path)
