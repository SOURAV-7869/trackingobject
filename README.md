# Object Tracking using OpenCV

This repository provides a Python script for performing object tracking using OpenCV. The script allows you to select a region of interest (ROI) in a video or webcam feed and track the selected object across frames.

## Features
- **ROI Selection**: Allows the user to select a region of interest (ROI) using the mouse.
- **Tracking with CamShift**: Tracks the selected object using the CamShift algorithm.
- **Supports Video or Webcam Input**: Can accept a video file as input or use the webcam for real-time tracking.
- **Dynamic ROI Selection**: User can interactively select the object to track by marking four points around the object.

## Requirements
- Python 3.x
- `numpy`
- `opencv-python`
- `argparse`

To install the required dependencies, you can use the following command:

```bash
pip install numpy opencv-python
```

## How to Use

1. **Clone the Repository**:
   Clone the repository to your local machine:

   ```bash
   git clone https://https://github.com/SOURAV-7869/trackingobject.git
   cd object-tracking
   ```

2. **Run the Script**:
   The script can be run with the following command:

   ```bash
   python track_object.py --video <video_file_path>
   ```

   If you don't provide a video file, the script will default to using your webcam.

   - Replace `<video_file_path>` with the path to your video file (optional). If you omit this argument, it will use the webcam (if available).
   - The script will display a window showing the current frame of the video or webcam.
   - Press the **"i"** key to enter ROI selection mode. Click on the four corners of the region to track.
   - Press the **"q"** key to quit the program at any time.

3. **Using Mouse for ROI Selection**:
   - Once you press the **"i"** key, you can select the ROI by clicking on four points around the object you want to track.
   - The program will automatically calculate the bounding box based on the selected points.
   - If the points are not selected or the coordinates are invalid, the program will give an error and allow for re-selection.

4. **Tracking**:
   - After selecting the ROI, the object will be tracked using the CamShift algorithm. The bounding box will be displayed around the tracked object as the video plays.

5. **Exiting the Program**:
   - Press the **"q"** key to exit the program.

## Code Explanation

### 1. `selectROI` function
This function handles the mouse input for selecting the region of interest (ROI). It listens for mouse events and records the selected points when the left mouse button is clicked. Once four points are selected, it calculates the bounding box for tracking.

### 2. `main` function
The `main` function handles the video capture (either from a webcam or video file) and the tracking loop. The following key tasks are performed:
   - **Video capture**: The script uses OpenCV’s `cv2.VideoCapture` to read video frames from either the webcam or a video file.
   - **ROI selection**: The user is prompted to select an ROI by clicking on the frame. This is handled by the `selectROI` callback.
   - **Tracking**: Once the ROI is selected, the script applies the CamShift tracking algorithm to track the selected object.
   - **Error handling**: The script includes error handling to ensure that the video feed is successfully captured and that the ROI coordinates are valid.

### 3. Key Presses
   - **"i"**: Enter ROI selection mode.
   - **"q"**: Exit the program.

## Troubleshooting
- **Webcam not working**: Ensure that your webcam is properly connected and accessible. Try using another application (e.g., Skype or Zoom) to check if the webcam is functional.
- **Error initializing video capture**: Make sure that the video file path is correct, or if using the webcam, ensure that no other application is using the camera.
- **Invalid ROI**: Ensure that exactly four points are selected for the ROI. The script will display an error message if there are fewer or more than four points.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
This script uses the OpenCV library, which provides various computer vision tools, including image and video manipulation, object tracking algorithms, and more. Visit the official [OpenCV documentation](https://docs.opencv.org/) for more information on its capabilities.
