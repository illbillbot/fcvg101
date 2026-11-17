# FCVG101 Task 3 — Visual Inspection Assistant

A Streamlit application that connects the existing coursework components:

- **Task 1A:** loads `calibration.npz` created by `calibration_task1a.py` and undistorts an uploaded image.
- **Task 1B:** calls the existing `src/classical_cv.py` helpers for spatial filtering, FFT filtering, Sobel/Canny edges, Harris corners, SIFT matching and RANSAC.
- **Task 2:** loads `artifacts/model.pth` through `src/inference.py` and classifies uploaded fruit images.

The original task folders remain separate. Task 1B's helper is loaded by absolute file path, while Task 2's inference module is imported from its configured project root to avoid collisions between separate packages named `src`.

## 1. Install and launch

Use a Python environment that has compatible versions of PyTorch and Torchvision for Task 2, plus the packages in this folder's `requirements.txt`.

From this Task 3 folder:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PyTorch and Torchvision are already installed in another working Python environment, run the installation and launch commands with that same interpreter (for example, `& $py -m pip install -r requirements.txt` and `& $py -m streamlit run app.py` in PowerShell, where `$py` is the full path to that interpreter).

## 2. Configure project paths

In the application's sidebar, enter the absolute paths to the three existing project folders:

1. **Task 1A:** folder containing `calibration_task1a.py`.
2. **Task 1B:** folder containing `src/classical_cv.py`.
3. **Task 2:** folder containing `src/inference.py`, `src/model.py`, `src/data_utils.py` and `artifacts/model.pth`.

The folder names and paths vary between machines. Do not assume the example paths in the code match your computer.

## 3. Available modes

### Fruit classification

Upload an image. The app loads the Task 2 checkpoint, applies the saved preprocessing and displays the predicted class and top softmax score. The score is not a calibrated probability of correctness.

### Camera calibration and undistortion

Upload an image captured by the calibrated camera at the same resolution as the calibration images. The app loads `K`, `dist`, `width` and `height` from a saved `.npz` file and runs `cv2.undistort`.

This mode consumes previously generated calibration parameters; it does not capture checkerboard images or execute `cv2.calibrateCamera` in the browser. The default expected file is `Task1A/results/real/calibration.npz`; use the optional sidebar calibration path if your file is elsewhere.

### Image filtering, FFT and edges

Upload one image and select Box, Gaussian, Median, ideal FFT low-pass/high-pass, Gaussian FFT low-pass, Sobel gradient magnitude, Canny edges or Harris corners.

### Feature detection and matching

Upload two views of the same scene with overlapping content. The app displays SIFT matches after Lowe's ratio test and reports RANSAC inliers.

## 4. Troubleshooting

- **Task 1B module not found:** check that the configured Task 1B directory contains `src/classical_cv.py`.
- **Task 2 module/checkpoint not found:** check that the configured Task 2 directory contains `src/inference.py`, `src/model.py`, `src/data_utils.py` and `artifacts/model.pth`.
- **Calibration file not found:** run Task 1A with calibration images and set the correct `.npz` path. This project only contains synthetic calibration views, so the generated parameters are not real-camera measurements.
- **Resolution mismatch:** undistortion requires an image at the calibration resolution unless the intrinsic matrix is appropriately scaled.
- **Torch/Torchvision import error:** run Streamlit in the same Python environment where a compatible PyTorch and Torchvision pair is installed.
- **SIFT unavailable:** use a compatible OpenCV build exposing `cv2.SIFT_create()`.

## 5. Limitations

- The dropdown explicitly routes the user's selected mode to a processing module; it is not a natural-language intent classifier.
- Calibration mode consumes saved Task 1A parameters rather than recalibrating in the app.
- Task 1A has synthetic-only calibration evidence; no physical-camera parameters were demonstrated.
- Task 1B matching evidence in the submitted app test uses a generated transformed scene pair. A real overlapping pair was not supplied.
- Task 2 metrics were measured on the selected Fruits-360 dataset and the project's custom split; they should not be generalized to arbitrary real-world photographs.
- The Streamlit interface reuses the Task 1B functions and Task 2 model instead of duplicating those algorithms.
