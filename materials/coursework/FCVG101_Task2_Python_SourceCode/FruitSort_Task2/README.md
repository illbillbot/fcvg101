# FCVG101 — Task 2: Fruit Classification

This project trains a PyTorch CNN to classify three Fruits-360 categories: **apple**, **banana** and **orange**.

## Repository contents

- `import_fruits360.py`: imports selected source folders into the three target classes.
- `prepare_dataset.py`: validates image files, removes exact duplicates, records a manifest and creates train/validation/test splits.
- `check_fruits360_split.py`: checks that generated group IDs do not cross the splits.
- `src/model.py`: FruitCNN architecture.
- `src/data_utils.py`: image transforms and dataset utilities.
- `train_balanced.py`: class-weighted model training and checkpoint selection.
- `evaluate.py`: held-out custom test-split evaluation.
- `predict.py`: single-image prediction.
- `artifacts/model.pth`: trained checkpoint used by Task 3.
- `artifacts/classes.json`: class mapping.
- `artifacts/*manifest*.csv`: import and split audit records.
- `results/`: training history, learning curves, test metrics, predictions and confusion matrix.

## Environment

Use Python with versions compatible with your installed PyTorch and Torchvision. The project was run with PyTorch 2.15.0+cpu and Torchvision 0.29.2+cpu in the author's environment.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you use an already-working Python environment with PyTorch and Torchvision, install the remaining requirements into that same interpreter.

## Dataset

The source is **Fruits-360 100x100**, credited to Mihai Oltean and licensed CC BY-SA 4.0. The dataset contains isolated fruit images on a mostly white background. The licence and upstream README are retained in `fruits-360-100x100-main/`.

The importer mapped selected apple, banana and orange source folders to three labels. Imported counts:

| Class | Images |
|---|---:|
| Apple | 15,792 |
| Banana | 719 |
| Orange | 1,338 |
| **Total** | **17,849** |

The dataset is imbalanced; class-weighted cross-entropy was therefore used during training.

## Reproduce the recorded run

From this Task 2 project directory:

```powershell
python import_fruits360.py --source .\fruits-360-100x100-main
python prepare_dataset.py
python check_fruits360_split.py
python train_balanced.py --epochs 15 --batch-size 32 --lr 0.001
python evaluate.py
```

The recorded training/test results in `results/` were produced with the dataset subset and split manifest stored in this project. Do not rerun dataset preparation or training if you intend to preserve the exact recorded split and checkpoint; doing so may overwrite the previous split or model.

For single-image inference:

```powershell
python predict.py --image .\data\splits\test\apple\YOUR_IMAGE.jpg
```

Replace `YOUR_IMAGE.jpg` with an existing image filename.

## Recorded experiment

- Train / validation / test images: 12,606 / 2,647 / 2,596.
- CNN: three convolutional stages (16, 32 and 64 channels), ReLU, max-pooling, adaptive average pooling, three output logits.
- Optimizer: Adam; learning rate 0.001; batch size 32; 15 epochs.
- Loss: inverse-frequency class-weighted cross-entropy.
- Best checkpoint: epoch 13, selected by validation macro F1 (weighted validation loss used as tie-break).
- Best validation accuracy: 99.32%.
- Best validation macro F1: 98.42%.
- Custom test split accuracy: **99.35%**.
- Custom test split macro F1: **0.9804**.

Test confusion matrix: 2,254 apples correctly classified, 9 apples predicted as banana, 8 apples predicted as orange, all 122 banana images correctly classified and all 203 orange images correctly classified.

## Evaluation caveats

1. The importer combines selected folders from both upstream `Training` and `Test` directories before making a new custom split. Therefore, these are **custom-split results**, not the official upstream test-set score.
2. Group IDs use a `frame_index // 4` rule. The audit confirms IDs do not cross the generated splits, but the 100x100 README does not independently confirm that this rule maps every group to one physical object. Treat physical-object independence as an assumption/limitation.
3. Near-duplicates are not comprehensively removed by exact-hash checking; perceptual/visual inspection is still recommended.
4. The white-background dataset is controlled imagery. Do not generalise the reported scores to arbitrary cluttered real-world photographs.
5. The model's top softmax output is a score, not a calibrated probability of correctness.

## Task 3 integration

Task 3 imports `src.inference.load_model` and `predict_pil`, and loads `artifacts/model.pth`. Keep the checkpoint, `src/model.py`, `src/data_utils.py`, `src/inference.py` and `artifacts/classes.json` together when moving the project.

## Attribution

Mihai Oltean, *Fruits-360 dataset*, https://github.com/fruits-360/fruits-360-100x100 — CC BY-SA 4.0. Attribute the source and preserve the ShareAlike terms when redistributing adapted dataset materials.
