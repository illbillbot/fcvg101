# FCVG101 Task 3 – Visual Inspection Assistant

A Streamlit application that connects the existing coursework components:

- **Task 1A:** loads `calibration.npz` created by `calibration_task1a.py` and undistorts an uploaded image.
- **Task 1B:** calls the existing `src/classical_cv.py` helpers for spatial filtering, FFT filtering, Sobel/Canny edges, Harris corners, SIFT matching and RANSAC.
- **Task 2:** loads `artifacts/model.pth` through the existing `src/inference.py` and classifies uploaded fruit images.

The original task folders remain separate. The app loads Task 1B's helper module by its file path and imports Task 2's `src.inference` from the configured Task 2 directory to avoid confusing the two `src` packages.

## 1. Expected project folders

Keep the existing folders and add this Task 3 folder alongside them:

```text
FCVG101/
├── Task1A_CameraCalibration/
│   └── results/real/calibration.npz
├── Task1B_ClassicalCV/
│   └── src/classical_cv.py
├── FruitSort_Task2/
│   ├── artifacts/model.pth
│   └── src/
│       ├── model.py
│       ├── data_utils.py
│       └── inference.py
└── Task3_VisualInspectionAssistant/
    ├── app.py
    ├── requirements.txt
    └── README.md
```

The folder names above are examples. The exact paths are configurable in the application's sidebar.

## 2. Install dependencies

It is fine for Python itself to be installed on `C:` while the coursework is on `D:`. Since Task 2 already has a working virtual environment with PyTorch and Torchvision, the simplest approach is to reuse that environment rather than downloading PyTorch again.

Activate the Task 2 environment (adjust the path if your Task 2 folder differs):

```powershell
& "D:\School\fcvg101\materials\CW\FCVG101_Task2_Python_SourceCode\FruitSort_Task2\.venv\Scripts\Activate.ps1"
python -m pip install streamlit opencv-python
```

If you prefer an isolated Task 3 environment, create one in this folder and install everything listed in `requirements.txt`. PyTorch and Torchvision may take additional time and disk space to install.

## 3. Configure and run

Open PowerShell in this Task 3 folder, then run:

```powershell
streamlit run app.py
```

In the left sidebar, enter the absolute paths to:

1. Task 1A folder (the one containing `calibration_task1a.py`).
2. Task 1B folder (the one containing `src/classical_cv.py`).
3. Task 2 folder (the one containing `src/inference.py` and `artifacts/model.pth`).

The default calibration location is `Task1A/results/real/calibration.npz`. If your real calibration output is stored elsewhere, enter its exact path in the optional calibration-file field.

## 4. Available modes

### Fruit classification

Upload an image. The app calls the Task 2 checkpoint and displays the predicted class and top softmax score. The score is not a calibrated probability.

### Camera calibration and undistortion

Upload an image captured by the calibrated camera at the same resolution as the calibration images. The app loads `K`, `dist`, `width` and `height` from the saved `.npz` file and runs `cv2.undistort`.

This mode consumes previously generated calibration parameters; it does not capture checkerboard images or run `cv2.calibrateCamera` in the web UI.

### Image filtering, FFT and edges

Upload one image and select Box, Gaussian, Median, ideal FFT low-pass/high-pass, Gaussian FFT low-pass, Sobel magnitude, Canny edges or Harris corners.

### Feature detection and matching

Upload two views of the same scene with overlapping content. The app displays SIFT matches after Lowe's ratio test and reports RANSAC inliers.

## 5. Troubleshooting

- **Task 1B module not found:** check that the configured Task 1B directory contains `src/classical_cv.py`.
- **Task 2 module/checkpoint not found:** check that the configured Task 2 directory contains `src/inference.py`, `src/model.py`, `src/data_utils.py` and `artifacts/model.pth`.
- **Calibration file not found:** run Task 1A with real checkerboard photographs and use the resulting `calibration.npz` path. Do not use synthetic calibration values as real camera measurements.
- **Resolution mismatch:** undistortion requires an image with the calibration image resolution unless the intrinsic matrix is appropriately scaled.
- **Torch/Torchvision import error:** use a compatible PyTorch/Torchvision installation in the environment running Streamlit.
- **SIFT unavailable:** install a compatible OpenCV build exposing `cv2.SIFT_create()`.

## 6. Limitations to state in the coursework report

- The mode selector explicitly routes the user's chosen task to the corresponding processing module; it is not a natural-language intent model.
- Calibration mode uses saved Task 1A parameters rather than recalibrating from a new checkerboard in the app.
- Fruit classification performance is measured on the selected Fruits-360 dataset and its custom split; it should not be presented as proof of equivalent performance on arbitrary real-world scenes.
- The Task 1B functions and Task 2 model are reused rather than reimplemented in the Streamlit interface.
