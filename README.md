# Multi-Angle Video Processor

This project processes multiple video files of the same scene from different angles and creates a single output video that selects the best frames based on pose estimation, text detection, and image quality. It also generates an editing script for manual refinement.

## Features

- Processes multiple input videos simultaneously
- Selects the best frame from each set of synchronized frames across all videos
- Uses MediaPipe Pose estimation to evaluate the visibility and centeredness of the subject
- Detects text in frames using Tesseract OCR
- Automatically adjusts brightness and contrast for consistency across angles
- Holds on the best angle for a specified duration
- Supports manual cuts via JSON configuration
- Generates editing scripts for manual refinement
- Handles videos with different frame rates through synchronization
- Automatically selects and incorporates audio from the most-used angle

## Requirements

- Python 3.6+
- OpenCV (cv2)
- MediaPipe
- NumPy
- tqdm
- pytesseract
- aspose.imaging
- ffmpeg (for audio processing)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/multi-angle-video-processor.git
   cd multi-angle-video-processor
   ```

2. Install the required packages:
   ```
   pip install opencv-python mediapipe numpy tqdm pytesseract aspose.imaging
   ```

3. Install system dependencies:
   - Tesseract OCR:
     - Ubuntu: `sudo apt-get install tesseract-ocr`
     - macOS: `brew install tesseract`
     - Windows: Download from [GitHub Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki)
   - FFmpeg:
     - Ubuntu: `sudo apt-get install ffmpeg`
     - macOS: `brew install ffmpeg`
     - Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)

## Usage

Run the script with the following arguments:

```
python video_processor.py <input_directory> <output_path> [--cuts_json CUTS_JSON] [--adjustment_mode {manual,auto,none}]
```

Arguments:
- `input_directory`: Path to directory containing input video files
- `output_path`: Path for the output video file
- `--cuts_json`: Optional path to JSON file containing predefined cuts
- `--adjustment_mode`: Choose between manual, automatic, or no brightness/contrast adjustment (default: auto)

## Output

The script generates:
1. A processed video file combining the best angles
2. A Python editing script in the `logs` directory for manual refinement
3. A JSON log file with angle usage statistics and timestamps

## How It Works

1. Video Synchronization:
   - Automatically detects and synchronizes videos with different frame rates
   - Uses the lowest FPS as the target to ensure smooth playback

2. Frame Selection:
   - Evaluates frames based on:
     - Pose visibility and centeredness (MediaPipe)
     - Text presence (Tesseract OCR)
     - Image quality (brightness/contrast)
   - Holds the best angle for a specified duration

3. Post-Processing:
   - Automatically adjusts brightness and contrast for consistency
   - Combines video with audio from the most-used angle
   - Generates an editing script for manual refinement

## Customization

You can adjust various parameters in the code:
- `hold_duration`: Duration to hold the best angle (in seconds)
- Scoring weights in the `compute_score` function
- Brightness and contrast adjustment parameters
- Frame synchronization settings