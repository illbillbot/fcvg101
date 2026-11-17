"""FCVG101 Task 3 - Visual Inspection Assistant.

This Streamlit application integrates the existing Task 1A calibration output,
Task 1B classical-CV helpers, and Task 2 fruit-classification model.

Run from this folder with:
    streamlit run app.py

Use the sidebar to enter the paths to the three existing coursework folders.
The folders remain separate; this app imports/loads their existing files.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps


st.set_page_config(
    page_title="Visual Inspection Assistant | FCVG101",
    page_icon="🔎",
    layout="wide",
)


def _default_path(env_name: str, relative_default: str) -> str:
    """Return an environment override or a path relative to this app folder."""
    import os

    value = os.environ.get(env_name)
    if value:
        return value
    return str((Path(__file__).resolve().parent / relative_default).resolve())


def _load_classical_module(task1b_dir: str) -> Any:
    """Load classical_cv.py by absolute path to avoid collisions between src packages."""
    source = Path(task1b_dir).expanduser().resolve() / "src" / "classical_cv.py"
    if not source.is_file():
        raise FileNotFoundError(f"Could not find Task 1B module: {source}")

    module_name = "fcvg_task1b_classical_cv"
    cached = sys.modules.get(module_name)
    if cached is not None and Path(getattr(cached, "__file__", "")).resolve() == source:
        return cached

    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_task2_inference(task2_dir: str) -> Any:
    """Import Task 2's src.inference from its own project directory.

    The Task 1B module is loaded under a distinct name above, so the two
    projects can safely retain their separate `src` directories.
    """
    root = Path(task2_dir).expanduser().resolve()
    if not (root / "src" / "inference.py").is_file():
        raise FileNotFoundError(f"Could not find Task 2 inference module in {root / 'src'}")

    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)

    # Ensure that a previously imported package named `src` does not point
    # at another coursework folder after a path is changed in the sidebar.
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]
    importlib.invalidate_caches()
    return importlib.import_module("src.inference")


def _uploaded_image_to_bgr(uploaded_file) -> np.ndarray:
    """Decode an uploaded image, respecting EXIF orientation."""
    with Image.open(uploaded_file) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


def _display_bgr(image: np.ndarray, caption: str | None = None) -> None:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(rgb, caption=caption, use_container_width=True)


def _normalise_for_display(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array
    finite = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return cv2.normalize(finite, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _get_calibration_file(task1a_dir: str, manual_path: str) -> Path:
    if manual_path.strip():
        candidate = Path(manual_path).expanduser()
    else:
        candidate = Path(task1a_dir).expanduser() / "results" / "real" / "calibration.npz"
    return candidate.resolve()


@st.cache_resource(show_spinner="Loading the trained fruit classifier...")
def _cached_model(task2_dir: str, checkpoint_path: str):
    inference = _load_task2_inference(task2_dir)
    return inference.load_model(checkpoint_path)


st.title("🔎 Visual Inspection Assistant")
st.caption("FCVG101 · Integrating camera calibration, classical computer vision and deep-learning classification")

with st.sidebar:
    st.header("Project configuration")
    st.caption("Enter the locations of your existing coursework folders. Keep Task 1A, Task 1B and Task 2 separate.")
    task1a_dir = st.text_input(
        "Task 1A folder",
        value=_default_path("FCVG_TASK1A_DIR", "../Task1A_CameraCalibration"),
        help="Folder containing calibration_task1a.py and its results directory.",
    )
    task1b_dir = st.text_input(
        "Task 1B folder",
        value=_default_path("FCVG_TASK1B_DIR", "../Task1B_ClassicalCV"),
        help="Folder containing src/classical_cv.py.",
    )
    task2_dir = st.text_input(
        "Task 2 folder",
        value=_default_path("FCVG_TASK2_DIR", "../FruitSort_Task2"),
        help="Folder containing src/inference.py and artifacts/model.pth.",
    )
    calibration_path = st.text_input(
        "Calibration file (optional)",
        value="",
        help="Leave blank to use Task 1A/results/real/calibration.npz. You can enter another .npz path here.",
    )
    st.divider()
    st.caption("Tip: use absolute Windows paths, for example D:\\School\\...\\FruitSort_Task2.")

mode = st.selectbox(
    "What would you like to inspect?",
    [
        "Fruit classification (Task 2)",
        "Camera calibration & undistortion (Task 1A)",
        "Image filtering, FFT & edges (Task 1B)",
        "Feature detection & matching (Task 1B)",
    ],
)

st.divider()

if mode == "Fruit classification (Task 2)":
    st.subheader("Fruit classification")
    st.write("Upload an apple, banana or orange image. The application loads your saved Task 2 checkpoint and applies the same preprocessing used during training.")
    uploaded = st.file_uploader("Choose a fruit image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="fruit_image")

    if uploaded is not None:
        col_image, col_result = st.columns([1, 1])
        with col_image:
            image_pil = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
            st.image(image_pil, caption="Uploaded image", use_container_width=True)
        with col_result:
            checkpoint = Path(task2_dir).expanduser() / "artifacts" / "model.pth"
            try:
                model, idx_to_class, transform = _cached_model(str(Path(task2_dir).expanduser().resolve()), str(checkpoint.resolve()))
                inference = _load_task2_inference(task2_dir)
                label, score = inference.predict_pil(model, idx_to_class, transform, image_pil)
                st.success(f"Prediction: {label.upper()}")
                st.metric("Top model score", f"{score:.3f}")
                st.caption("This is the highest softmax score, not a calibrated probability of correctness.")
            except Exception as exc:
                st.error(f"Could not run the fruit classifier: {exc}")
                st.info("Check the Task 2 folder path and confirm that artifacts/model.pth exists.")

elif mode == "Camera calibration & undistortion (Task 1A)":
    st.subheader("Camera calibration and image undistortion")
    st.write("This mode uses the calibration parameters saved by your Task 1A script. It does not perform checkerboard calibration inside the browser; run calibration_task1a.py first to create calibration.npz.")
    uploaded = st.file_uploader("Upload an image captured with the calibrated camera", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], key="calibration_image")
    if uploaded is not None:
        calibration_file = _get_calibration_file(task1a_dir, calibration_path)
        try:
            if not calibration_file.is_file():
                raise FileNotFoundError(f"Calibration file not found: {calibration_file}")
            with np.load(calibration_file, allow_pickle=False) as saved:
                K = saved["K"]
                dist = saved["dist"]
                expected_width = int(saved["width"])
                expected_height = int(saved["height"])
            image_bgr = _uploaded_image_to_bgr(uploaded)
            height, width = image_bgr.shape[:2]
            if (width, height) != (expected_width, expected_height):
                raise ValueError(
                    f"Image resolution is {width}x{height}; calibration expects "
                    f"{expected_width}x{expected_height}. Use an image from the same camera/resolution."
                )
            corrected = cv2.undistort(image_bgr, K, dist)
            left, right = st.columns(2)
            with left:
                _display_bgr(image_bgr, "Original image")
            with right:
                _display_bgr(corrected, "Undistorted image")
            with st.expander("Camera parameters"):
                st.write("Intrinsic matrix K")
                st.code(np.array2string(K, precision=5, suppress_small=True))
                st.write("Distortion coefficients")
                st.code(np.array2string(dist.reshape(-1), precision=6, suppress_small=True))
                st.caption(f"Loaded calibration: {calibration_file}")
            st.download_button("Download undistorted image", cv2.imencode(".png", corrected)[1].tobytes(), "undistorted.png", "image/png")
        except Exception as exc:
            st.error(f"Could not undistort image: {exc}")
            st.info("Check the calibration path, and confirm that you calibrated using real images from the same camera and resolution.")

elif mode == "Image filtering, FFT & edges (Task 1B)":
    st.subheader("Classical image processing")
    st.write("Select a processing operation from your Task 1B implementation.")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="classical_image")
    operation = st.selectbox(
        "Processing operation",
        ["Box filter", "Gaussian filter", "Median filter", "FFT ideal low-pass", "FFT ideal high-pass", "FFT Gaussian low-pass", "Sobel gradient magnitude", "Canny edges", "Harris corners"],
    )
    kernel = st.select_slider("Filter kernel size", options=[3, 5, 7, 9], value=5)
    sigma = st.slider("Gaussian sigma", min_value=0.2, max_value=4.0, value=1.2, step=0.1)
    cutoff = st.slider("FFT cutoff radius", min_value=5, max_value=100, value=26, step=1)

    if uploaded is not None:
        try:
            cv = _load_classical_module(task1b_dir)
            bgr = _uploaded_image_to_bgr(uploaded)
            gray = cv.to_gray(bgr)
            output: np.ndarray
            is_gray = True
            metrics_text = ""

            if operation in {"Box filter", "Gaussian filter", "Median filter"}:
                filtered = cv.filtering_outputs(gray, kernel=kernel, sigma=sigma)
                key = {"Box filter": "Box", "Gaussian filter": "Gaussian", "Median filter": "Median"}[operation]
                output = filtered[key]
            elif operation.startswith("FFT"):
                freq = cv.frequency_outputs(gray, cutoff=cutoff, gauss_sigma=max(float(cutoff), 1.0))
                key = {
                    "FFT ideal low-pass": "ideal_low_img",
                    "FFT ideal high-pass": "ideal_high_img",
                    "FFT Gaussian low-pass": "gaussian_low_img",
                }[operation]
                output = _normalise_for_display(freq[key])
            elif operation == "Sobel gradient magnitude":
                edge_results = cv.edges_outputs(gray, blur_kernel=kernel, sigma=sigma)
                output = _normalise_for_display(edge_results["magnitude"])
            elif operation == "Canny edges":
                edge_results = cv.edges_outputs(gray, blur_kernel=kernel, sigma=sigma)
                output = edge_results["canny"]
            else:
                response, points = cv.harris_points(gray)
                output = cv.draw_harris(bgr, points)
                is_gray = False
                metrics_text = f"Harris interest points detected: {len(points)}"

            left, right = st.columns(2)
            with left:
                _display_bgr(bgr, "Original image")
            with right:
                if is_gray:
                    st.image(output, caption=operation, clamp=True, use_container_width=True)
                else:
                    _display_bgr(output, operation)
            if metrics_text:
                st.info(metrics_text)
            encoded = cv2.imencode(".png", output)[1].tobytes()
            st.download_button("Download processed result", encoded, "processed_result.png", "image/png")
        except Exception as exc:
            st.error(f"Could not run this Task 1B operation: {exc}")
            st.info("Check that the Task 1B folder contains src/classical_cv.py and its Python dependencies are installed.")

else:  # Feature detection and matching
    st.subheader("SIFT feature detection and matching")
    st.write("Upload two views of the same scene with overlapping content. The application applies SIFT, Lowe's ratio test and RANSAC homography verification.")
    col_a, col_b = st.columns(2)
    with col_a:
        first_upload = st.file_uploader("First image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="match_first")
    with col_b:
        second_upload = st.file_uploader("Second image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="match_second")
    ratio = st.slider("Lowe ratio threshold", min_value=0.50, max_value=0.95, value=0.75, step=0.01)

    if first_upload is not None and second_upload is not None:
        try:
            cv = _load_classical_module(task1b_dir)
            image_a = _uploaded_image_to_bgr(first_upload)
            image_b = _uploaded_image_to_bgr(second_upload)
            kp_a, kp_b, good_matches = cv.sift_match(image_a, image_b, ratio=ratio)
            homography, inliers = cv.homography_inliers(kp_a, kp_b, good_matches)
            shown_matches = good_matches[:100]
            visual = cv2.drawMatches(
                image_a, kp_a, image_b, kp_b, shown_matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            st.image(cv2.cvtColor(visual, cv2.COLOR_BGR2RGB), caption="SIFT correspondences (up to 100 displayed)", use_container_width=True)
            a, b, c = st.columns(3)
            a.metric("Keypoints · image 1", len(kp_a))
            b.metric("Keypoints · image 2", len(kp_b))
            c.metric("Lowe ratio-test matches", len(good_matches))
            st.metric("RANSAC inliers", inliers)
            if homography is None:
                st.warning("A homography could not be estimated. Use images with more overlap and distinctive features.")
            else:
                with st.expander("Estimated homography matrix"):
                    st.code(np.array2string(homography, precision=5, suppress_small=True))
            st.download_button("Download matching visualisation", cv2.imencode(".png", visual)[1].tobytes(), "sift_matches.png", "image/png")
        except Exception as exc:
            st.error(f"Could not run feature matching: {exc}")
            st.info("SIFT requires an OpenCV build that exposes cv2.SIFT_create(). Check the Task 1B folder and OpenCV installation.")

st.divider()
st.caption("FCVG101 coursework prototype · Results depend on the quality and provenance of the uploaded images and calibration data.")
