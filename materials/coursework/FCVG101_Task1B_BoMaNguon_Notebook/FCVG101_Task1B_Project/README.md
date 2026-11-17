# FCVG101 - Task 1B (Spatial Filtering, FFT, Edges, Features)

## Mục tiêu
Bộ mã nguồn giúp thực hành Task 1B với dữ liệu minh hoạ tự sinh, gồm:
1. Gaussian và Salt-and-Pepper noise; Box/Gaussian/Median; MSE/PSNR.
2. FFT, fftshift, ideal/Gaussian low-pass, high-pass, inverse FFT.
3. Sobel Gx/Gy/gradient magnitude và Canny.
4. Harris corner response, SIFT descriptor, Lowe ratio test và RANSAC kiểm tra hình học (nâng cao).

## Chạy bằng Terminal
Tại thư mục dự án chứa file `demo_task1b.py`:

```bash
python -m pip install -r requirements.txt
python demo_task1b.py
```

Thư mục `results/` sẽ chứa hình PNG, `metrics_filtering.csv`, `metrics_summary.json`.

## Chạy bằng Jupyter Notebook
Mở terminal tại thư mục dự án, chạy:

```bash
jupyter notebook
```

Mở `notebooks/FCVG101_Task1B_Practice.ipynb`, chọn `Run All`.
Lưu ý: notebook dùng đường dẫn tương đối, thư mục làm việc phải là thư mục dự án.

## Thay ảnh mẫu bằng ảnh thật
- Giữ ảnh gốc chưa bị biến đổi để tạo noise kiểm soát và tính MSE/PSNR.
- Thay `base = create_demo_scene()` trong demo bằng `base = load_bgr('data/my_image.jpg')`.
- Đối với matching: cần hai ảnh thật có vùng chồng lấp. `same_scene_transformed.png` chỉ phục vụ test.
- Không ghi số demo là số liệu camera/dataset thực tế; trích dẫn nguồn ảnh và bản quyền ảnh thực.

## Lưu ý về Jupyter
Code không dùng `argparse.parse_args()` nên tránh được lỗi `ipykernel_launcher.py: unrecognized arguments -f`.

## Phân biệt yêu cầu và mở rộng
Theo hướng dẫn Coursework: filtering, FFT, gradients, Harris/SIFT/ORB, Lowe ratio test và bằng chứng hình/số liệu. RANSAC là phần tăng cường kiểm tra độ đúng hình học, không phải ngưỡng bắt buộc được nêu trong tài liệu hướng dẫn.
