# Stereo Vision Project: Detection, Calibration, and Position Calculation of an Object Using Two Cameras

## Overview
This project aims to detect, calibrate, and calculate the position of a known object using a stereo vision system with two cameras. The project is divided into four main parts, each addressing a specific aspect of the stereo vision workflow. The enhancements include object tracking with route tracing and multithreading for optimized performance. Below are the details:

---

## Project Structure

### Main Components:
1. **Object Detection**
    - **Script**: `train_yolov9_object_detection_on_custom_dataset.ipynb`
    - **Description**: Contains the training steps for the YOLOv9 object detector using a custom dataset.
    - **Dataset**: The deodorant detection dataset is available at [Roboflow](https://universe.roboflow.com/computer-vision-muvdl/deodorant-detection).

2. **Camera Calibration**
    - Develops intrinsic and extrinsic parameter matrices for the stereo camera setup.
    - Explains the calibration method in the report.

3. **Position and Distance Calculation**
    - **Main Script**: `part3_script.py`
    - **Interface**: `part3_interface.py` (developed using Streamlit).
    - Calculates the 3D position of an object and midpoints between camera projections.
    - Outputs real-time coordinates and visualizations.

4. **Enhancements**
    - **Object Tracking**: Includes route tracing, speed calculation, and next-position prediction.
    - **Multithreading**: Ensures optimized execution of object detection and tracking processes.

---

## Folder Structure

```
Rapport_vision.pdf/
train_yolov9_object_detection_on_custom_dataset.ipynb/
object_detection_app/
├── yolov9/
│   ├── requirements.txt
│   ├── steps_explanation.txt
│   ├── execute.txt
│   ├── detect.py
│   ├── detect_tracking_old.py
├── /
│   ├── part3_script.py
│   ├── part3_interface.py
```

---

## How to Execute

### Prerequisites
1. Install Python 3.9+.
2. Ensure you have access to the required hardware (two cameras).

### Steps
1. **Set up the environment**:
    ```bash
    python -m venv stereo_env
    source stereo_env/bin/activate  # On Windows: stereo_env\Scripts\activate
    ```
2. **Install requirements**:
    ```bash
    pip install -r yolov9/requirements.txt
    ```
3. **Run the Interface**:
    ```bash
    streamlit run object_detection_app/part3_interface.py
    ```
4. Follow the on-screen instructions to perform detection, calibration, and position calculation.

### YOLOv9 Usage Steps
- Refer to `steps_explanation.txt` for details on using `detect.py` and `object_tracking.py`.
- Use `execute.txt` for specific execution commands.

---

## Methodology

### Part 1: Object Detection
- **Method**: The YOLOv9 model is trained on a custom deodorant dataset.
- **Output**: The position (x, y) or center of the detected object in the image frame.

### Part 2: Camera Calibration
- **Technique**: Checkerboard calibration to generate intrinsic and extrinsic parameter matrices.
- **Output**: Accurate camera parameters for stereo vision.

### Part 3: Position and Distance Calculation
- **Setup**: Two cameras with horizontal disparity.
- **Steps**:
  1. Calibrate cameras (Part 2).
  2. Detect object positions in both images (Part 1).
  3. Compute 3D coordinates using stereo triangulation.
  4. Display real-time coordinates and midpoint projections.

### Part 4: Enhancements
- **Object Tracking**:
  - Tracks the route of the object.
  - Calculates speed and predicts future positions.
- **Multithreading**: Parallelizes detection ,tracking , calibration and all processes for real-time performance.

---

## Additional Notes
- The project includes comprehensive comments in the code.
- If questions arise, feel free to raise them for clarification.

---

## Contributors
- Mohamed Anes Mihoubi
- Kenza Taouci
- Hamza Taourirt
- Abderahmane Aitidir

---

## Acknowledgments
This project integrates prior work on object detection and tracking, including the following:
- YOLOv9 implementation on a deodorant dataset.
- Custom calibration scripts.
- depth estimation.
- calculation of object-to-camera distance, and determining the object's position in real-world coordinates.

---

Feel free to explore, adapt, and expand upon this project for further stereo vision applications.

