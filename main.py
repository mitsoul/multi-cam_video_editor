import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm
import argparse
import pytesseract

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process videos to select the best frames based on pose.")
    parser.add_argument('input_dir', type=str, help='Path to the input video directory')
    parser.add_argument('output_path', type=str, help='Path to the output video file')
    return parser.parse_args()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def detect_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return len(text.strip()) > 0

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

# Function to compute combined score
def compute_score(landmarks, frame_width, frame_height):
    visibility = visibility_score(landmarks)
    centeredness = centeredness_score(landmarks, frame_width, frame_height)
    return visibility * 0.4 + centeredness * 0.6  # Adjust weights if needed

# Process the videos
def process_videos(video_dir, output_path):
    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]
    video_caps = [cv2.VideoCapture(vf) for vf in video_files]
    
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in video_caps]
    frame_widths = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in video_caps]
    frame_heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in video_caps]
    fps = video_caps[0].get(cv2.CAP_PROP_FPS)  # Assuming same FPS for all videos

    # Prepare output video writer
    output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_widths[0], frame_heights[0]))

    # Track progress with tqdm
    total_frames = min(frame_counts)
    frame_duration = 1 / fps  # Duration of each frame in seconds
    hold_duration = 1  # Duration to hold the best angle in seconds
    hold_frames = int(hold_duration / frame_duration)  # Number of frames 

    with tqdm(total=total_frames, desc="Processing Frames", unit="frame", ncols=100) as pbar:
        # Iterate over frames
        hold_counter = 0
        best_cap_idx = None
        

        for i in range(total_frames):  # Loop through synced frames
            if hold_counter > 0:
                # Continue using frames from the best video
                ret, frame = video_caps[best_cap_idx].read()
                if ret:
                    output_video.write(frame)
                hold_counter -= 1
            else:
                best_score = -float('inf')
                text_detected = False
                for cap_idx, cap in enumerate(video_caps):
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame_rgb)

                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        score = compute_score(landmarks, frame_widths[cap_idx], frame_heights[cap_idx])

                        if score > best_score:
                            best_score = score
                            best_cap_idx = cap_idx
                
                    if detect_text(frame):
                        text_detected = True
                        best_cap_idx = cap_idx
                        break  # Exit the loop early if text is detected

                # Write the best frame to the output video
                if best_cap_idx is not None:
                    video_caps[best_cap_idx].set(cv2.CAP_PROP_POS_FRAMES, i)  # Reset to the correct frame
                    ret, frame = video_caps[best_cap_idx].read()
                    if ret:
                        output_video.write(frame)
                        hold_counter = hold_frames if text_detected else hold_frames - 1  # Adjust hold counter

            # Update tqdm progress bar
            pbar.update(1)

    # Release all resources
    output_video.release()
    for cap in video_caps:
        cap.release()

if __name__ == "__main__":
    args = parse_arguments()
    process_videos(args.input_dir, args.output_path)
