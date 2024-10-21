# Multi-Angle Video Processor

This project processes multiple video files of the same scene from different angles and creates a single output video that selects the best frames based on pose estimation, text detection, and image quality.

## Features

- Processes multiple input videos simultaneously
- Selects the best frame from each set of synchronized frames across all videos
- Uses MediaPipe Pose estimation to evaluate the visibility and centeredness of the subject
- Detects text in frames using Tesseract OCR
- Adjusts brightness and contrast of frames for consistency
- Holds on the best angle for a specified duration
- Provides a progress bar for processing feedback

## Requirements

- Python 3.6+
- OpenCV (cv2)
- MediaPipe
- NumPy
- tqdm
- pytesseract

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/multi-angle-video-processor.git
   cd multi-angle-video-processor
   ```

2. Install the required packages:
   ```
   pip install opencv-python mediapipe numpy tqdm pytesseract
   ```

3. Install Tesseract OCR on your system:
   - For Ubuntu: `sudo apt-get install tesseract-ocr`
   - For macOS: `brew install tesseract`
   - For Windows: Download and install from [GitHub Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage

Run the script from the command line with the following arguments:

```
python main.py <input_directory> <output_path>
```

## How It Works

1. The script reads all .mp4 files from the input directory.
2. It processes frames from all videos simultaneously, selecting the best frame based on:
   - Pose visibility and centeredness (using MediaPipe Pose)
   - Presence of text (using Tesseract OCR)
   - Consistent brightness and contrast
3. The selected frames are written to the output video.
4. The best angle is held for a specified duration before checking for a better angle.

## Customization

You can adjust the following parameters in the script:

- `hold_duration`: Duration to hold the best angle (in seconds)
- Weights for visibility and centeredness scores in the `compute_score` function
- Threshold for considering a landmark visible in the `visibility_score` function
