# FCVG101 — Computer Vision Coursework

**Visual Inspection Assistant: classical image analysis and three-class fruit classification**

This repository contains the code and supporting materials for the FCVG101 Introduction to Computer Vision coursework.

## Project structure

The coursework source folders are kept separate under `materials/coursework/`:

- **Task 1A — Camera calibration:** `FCVG101_Task1A_BoMaNguon_ThucHanh/FCVG101_Task1A_Workshop/`
- **Task 1B — Classical computer vision:** `FCVG101_Task1B_BoMaNguon_Notebook/FCVG101_Task1B_Project/`
- **Task 2 — Fruit classification:** `FCVG101_Task2_Python_SourceCode/FruitSort_Task2/`
- **Task 3 — Streamlit assistant:** `task3_visual_inspection_assistant/task3_visual_inspection_assistant/`

The original coursework brief and workshop resources are also under `materials/coursework/`.

## Task 1A — Camera calibration

The implementation uses OpenCV checkerboard detection, `cv2.calibrateCamera`, reprojection-error calculations and `cv2.undistort`.

**Evidence limitation:** the supplied calibration images are generated synthetic views. No photographs were captured using a physical camera, so the reported intrinsic matrix, distortion coefficients and reprojection errors are synthetic pipeline smoke-test outputs only. They are not measurements of a real camera.

To run the synthetic demonstration, open a terminal in the Task 1A project folder and run:

```bash
python -m pip install -r requirements.txt
python generate_synthetic_checkerboards.py
python calibration_task1a.py --input data/calibration --output results/demo --cols 9 --rows 6 --square-mm 18
```

## Task 1B — Classical computer vision

The reusable `src/classical_cv.py` module and demonstration scripts cover:

- Box, Gaussian and median filtering
- Controlled Gaussian and salt-and-pepper noise, MSE and PSNR
- 2D FFT frequency filtering (ideal low-pass/high-pass and Gaussian low-pass)
- Sobel gradients and Canny edges
- Harris interest points
- SIFT descriptors, Lowe's ratio test and RANSAC homography estimation

Run the demonstration from the Task 1B project folder:

```bash
python -m pip install -r requirements.txt
python demo_task1b.py
```

The provided matching pair is synthetic. The browser integration test reported 252 keypoints in image 1, 331 in image 2, 140 Lowe ratio-test matches and 126 RANSAC inliers (90.0%). These are results for the synthetic pair, not a real-world matching benchmark.

The standalone image `data/my_image.jpg` was used for application-level filter/edge tests. Its original source and licence should be confirmed by the project author.

## Task 2 — Deep-learning fruit classifier

The classifier has three target labels: **apple, banana and orange**. The model is a compact three-stage CNN implemented in PyTorch.

### Reported experiment

- Imported images: 17,849 (apple 15,792; banana 719; orange 1,338)
- Train / validation / test: 12,606 / 2,647 / 2,596 images
- Training: 15 epochs, batch size 32, Adam, learning rate 0.001, class-weighted cross-entropy
- Best checkpoint: epoch 13, selected using validation macro F1
- Custom test split accuracy: **99.35%**
- Custom test split macro F1: **0.9804**

The test score is for the project's custom split, not the original upstream Fruits-360 test partition. The importer combined selected folders from the upstream Training and Test directories before making a new split. The frame-index grouping rule is an assumption for this 100x100 release; the audit verifies that generated group IDs do not cross the created splits but cannot independently prove physical-fruit identity. Treat the metrics as controlled-dataset results, not expected performance on arbitrary photographs.

The dataset images have been isolated against white backgrounds. The upstream dataset is licensed CC BY-SA 4.0; retain attribution and share-alike terms. See the dataset README and licence in `fruits-360-100x100-main/`.

### Reproducing Task 2

1. Install the dependencies listed in Task 2's `requirements.txt`.
2. Download the Fruits-360 100x100 dataset from the upstream project: https://github.com/fruits-360/fruits-360-100x100
3. Extract it and run the importer with the extracted dataset directory as the source, for example:

```bash
python import_fruits360.py --source /path/to/fruits-360-100x100-main
python prepare_dataset.py
python train_balanced.py --epochs 15 --batch-size 32 --lr 0.001
python evaluate.py
```

The repository includes the trained checkpoint at `artifacts/model.pth`, class metadata and experiment result files. It also includes import/split manifests. The full dataset is large and should normally be obtained from the upstream source rather than redistributed from this repository.

To make a single-image prediction after preparing the checkpoint and environment:

```bash
python predict.py --image /path/to/image.jpg
```

## Task 3 — Visual Inspection Assistant

The Streamlit interface integrates the Task 1A calibration output, Task 1B processing functions and Task 2 classifier.

Run the app from its own folder:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Use the sidebar to set the absolute paths to the Task 1A, Task 1B and Task 2 project folders on your machine. The app's default relative paths are examples and may need to be changed for your folder layout.

The fruit-classification mode loads Task 2's `artifacts/model.pth`. Calibration/undistortion mode expects a saved `calibration.npz` made by the Task 1A script and matching image dimensions. The app does not calibrate a camera inside the browser. Since this project has synthetic-only calibration data, the real-camera calibration mode has not been demonstrated with physical camera parameters.

### Application checks recorded

- Apple, orange and banana sample images were classified correctly in the Streamlit interface (top softmax scores 0.99, 1.00 and 1.00).
- Box, Gaussian and median filtering were displayed.
- Canny, Sobel magnitude, ideal FFT low-pass/high-pass and Harris visualisations were displayed.
- SIFT/RANSAC was tested on the synthetic transformed scene pair.

A top softmax score is a model score, not a calibrated probability of correctness.

## Important limitations

1. **Task 1A:** only synthetic checkerboard views were used; no real-camera calibration experiment was performed.
2. **Task 1B:** the demonstrated SIFT/RANSAC correspondence pair is synthetic; a real overlapping image pair was not provided.
3. **Task 2:** the evaluation uses a custom split and a controlled white-background dataset; physical-object grouping for the 100x100 release is not independently verified.
4. **Image attribution:** the author should add source/licence information for `my_image.jpg` before redistributing it publicly, or remove/replace it if permission cannot be established.
5. **Environment:** do not commit virtual environments, API keys, personal files or unnecessary cache directories.

## Submission report

The final coursework report is a separate deliverable. Add the submitted PDF to the repository only if your course rules permit making it public. Otherwise, submit it separately through the course CMS and share the repository link with the marker.

## Credits

- Coursework requirements: FCVG101 Introduction to Computer Vision assignment brief.
- Fruit dataset: Mihai Oltean, Fruits-360, https://github.com/fruits-360/fruits-360-100x100 (CC BY-SA 4.0).
- Implementation tools: OpenCV, NumPy, PyTorch, Torchvision and Streamlit.
