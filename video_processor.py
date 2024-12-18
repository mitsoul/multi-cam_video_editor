import cv2
import mediapipe as mp
import numpy as np
import json
import os
from tqdm import tqdm
import argparse
from frame_adjustments import *
from utils import *

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process videos to select the best frames based on pose.")
    parser.add_argument('input_dir', type=str, help='Path to the input video directory')
    parser.add_argument('output_path', type=str, help='Path to the output video file')
    parser.add_argument('--cuts_json', type=str, help='Path to JSON file containing predefined cuts (optional)')
    parser.add_argument('--adjustment_mode', type=str, choices=['manual', 'auto', 'none'], 
                       default='auto', help='Choose between manual, automatic, or no brightness/contrast adjustment')
    return parser.parse_args()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

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
def compute_score(landmarks, frame_width, frame_height, frame):
    visibility = visibility_score(landmarks)
    centeredness = centeredness_score(landmarks, frame_width, frame_height)
    brightness_score = calculate_brightness_score(frame)
    text_score = detect_text(frame) / 100  # Normalize text score (assuming 100 characters is a good benchmark)
    return visibility * 0.5 + centeredness * 0.2  + text_score * 1.2

def generate_editing_script(angle_timestamps, video_files, output_path):
        # Extract lecture number from the first video filename
        import re
        first_file = os.path.basename(video_files[0])
        match = re.search(r'(\d+)\.701', first_file)
        lec_num = int(match.group(1)) if match else 1

        # Map video indices to their types based on filename
        angle_types = {}
        for idx, file in enumerate(video_files):
            filename = os.path.basename(file).lower()
            if 'left' in filename:
                angle_types[idx] = 'left'
            elif 'center' in filename:
                angle_types[idx] = 'center'
            elif 'wide' in filename:
                angle_types[idx] = 'wide'
            elif 'tracking' in filename:
                angle_types[idx] = 'tracking'
            elif 'pc' in filename:
                angle_types[idx] = 'pc'

        script = [
            f"lec_num = {lec_num}",
            "prefix = \"videos/\"",
            f"left_source = prefix + f\"dcai_lec{{lec_num:02d}}_left.mp4\"",
            f"center_source = prefix + f\"dcai_lec{{lec_num:02d}}_center.mp4\"",
            f"wide_source = prefix + f\"dcai_lec{{lec_num:02d}}_wide.mp4\"",
            f"tracking_source = prefix + f\"dcai_lec{{lec_num:02d}}_tracking.mp4\"",
            f"pc_source = prefix + f\"dcai_lec{{lec_num:02d}}_pc.mp4\"",
            f"audio_source = prefix + f\"dcai_lec{{lec_num:02d}}_tracking.mp4\"",
            "",
            "left = Fullscreen(left_source, delay=-17/30)",
            "center = Fullscreen(center_source, delay=-12/30)",
            "pc = Fullscreen(pc_source, delay=-12/30)",
            "wide = Fullscreen(wide_source, delay=1/30)",
            "tracking = Fullscreen(tracking_source, delay=0/30)",
            "pc_and_tracking = Overlay(pc, tracking, crop_x = 0, crop_y = 0, crop_width = 1920, location = Location.TOP_RIGHT, width = 270, margin = 10)",
            "audio = Audio(audio_source, delay=1/30)",
            "",
            "Multitrack([",
        ]

        # Add clips based on angle_timestamps
        clips = []
        for segment in angle_timestamps:
            angle = segment['angle']
            if angle in angle_types:
                angle_name = angle_types[angle]
                source = angle_name if angle_name != 'pc' else 'pc_and_tracking'
                clips.append(f"    Clip({source}, start=\"{segment['start']}\"{', end=\"' + segment['end'] + '\"' if segment == angle_timestamps[-1] else ''})")

        script.extend(clips)
        script.extend([
            "    ], audio).render(f\"dcai_lec{lec_num:02d}.mp4\")"
        ])

        return "\n".join(script)

# Process the videos
def process_videos(video_dir, output_path, cuts_json=None, adjustment_mode='auto'):
    # Add timestamp tracking
    angle_timestamps = []
    current_time = 0.0
    segment_start = 0.0  # Add this at the beginning
    current_angle = None  # Add this to track the current angle

    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]
    video_caps = [cv2.VideoCapture(vf) for vf in video_files]

    # Update script output path to use video_editor/logs directory
    script_output_path = os.path.join('logs', os.path.basename(output_path).replace('.mp4', '_script.py'))
    os.makedirs(os.path.dirname(script_output_path), exist_ok=True)  # Ensure directory exists

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

    adjustments = calculate_target_variables(sample_frames)
    
    if cuts_json:
        print("Creating output video using provided cuts.")
        with open(cuts_json, 'r') as f:
            cuts_data = json.load(f)
            angle_segments = cuts_data.get('angle_segments', [])
    
        with tqdm(total=total_frames, desc="Processing Frames", unit="frame", ncols=100) as pbar:
            frame_idx = 0
            current_segment_idx = 0
            
            while frame_idx < total_frames and current_segment_idx < len(angle_segments):
                segment = angle_segments[current_segment_idx]
                current_time = frame_idx / target_fps
                segment_start = parse_timestamp(segment['start'])
                segment_end = parse_timestamp(segment['end'])
                angle_idx = segment['angle']

                # Add timestamp tracking for cuts_json mode
                if current_angle != angle_idx:
                    if current_angle is not None:
                        angle_timestamps.append({
                            'angle': current_angle,
                            'start': format_timestamp(segment_start),
                            'end': format_timestamp(current_time),
                            'video_file': os.path.basename(video_files[current_angle])
                        })
                    current_angle = angle_idx
                    segment_start = current_time

                if current_time >= segment_start and current_time < segment_end:
                    frames = read_synchronized_frames()
                    if frames:
                        for cap_idx, frame in frames:
                            if cap_idx == angle_idx:
                                brightness = adjustments[cap_idx]['brightness']
                                contrast = adjustments[cap_idx]['contrast']
                                adjusted_frame = adjust_frame(frame, brightness + 100, int(contrast * 100))
                                output_video.write(adjusted_frame)
                                angle_usage_count[angle_idx] += 1
                    frame_idx += 1
                    pbar.update(1)
                else:
                    current_segment_idx += 1
    
    else:
        print("Creating output video")
        with tqdm(total=total_frames, desc="Processing Frames", unit="frame", ncols=100) as pbar:
            hold_counter = 0
            best_cap_idx = None
            
            for i in range(total_frames):
                current_time = i / target_fps

                if hold_counter > 0:
                    frames = read_synchronized_frames()
                    if frames:
                        for cap_idx, frame in frames:
                            if cap_idx == best_cap_idx:
                                brightness = adjustments[cap_idx]['brightness']
                                contrast = adjustments[cap_idx]['contrast']
                                adjusted_frame = adjust_frame(frame, brightness + 100, int(contrast * 100))
                                output_video.write(adjusted_frame)
                                angle_usage_count[best_cap_idx] += 1
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

                        if best_frame is not None:
                        # Add timestamp tracking for automatic mode
                            if current_angle != best_cap_idx:
                                if current_angle is not None:
                                    angle_timestamps.append({
                                        'angle': current_angle,
                                        'start': format_timestamp(segment_start),
                                        'end': format_timestamp(current_time),
                                        'video_file': os.path.basename(video_files[current_angle])
                                    })
                                current_angle = best_cap_idx
                                segment_start = current_time

                            brightness = adjustments[best_cap_idx]['brightness']
                            contrast = adjustments[best_cap_idx]['contrast']
                            adjusted_frame = adjust_frame(best_frame, brightness + 100, int(contrast * 100))
                            output_video.write(adjusted_frame)
                            angle_usage_count[best_cap_idx] += 1
                            hold_counter = hold_frames - 1

                pbar.update(1)
            
            # # Log the final segment
            # if best_cap_idx is not None:
            #     angle_timestamps.append({
            #         'angle': best_cap_idx,
            #         'start': format_timestamp(segment_start),
            #         'end': format_timestamp(current_time),
            #         'video_file': os.path.basename(video_files[best_cap_idx])
            #     })

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

    # Add the final segment before finishing
    if current_angle is not None:
        angle_timestamps.append({
            'angle': current_angle,
            'start': format_timestamp(segment_start),
            'end': format_timestamp(current_time),
            'video_file': os.path.basename(video_files[current_angle])
        })

    script_content = generate_editing_script(angle_timestamps, video_files, output_path)
    with open(script_output_path, 'w') as f:
        f.write(script_content)
    
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)

if __name__ == "__main__":
    args = parse_arguments()
    process_videos(args.input_dir, args.output_path, args.cuts_json, args.adjustment_mode)