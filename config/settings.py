from pathlib import Path

from torch import tensor

# --- プロジェクトのベースパス ---
BASE_DIR = Path(__file__).resolve().parent.parent

# --- データ関連のパス ---
DATA_DIR = BASE_DIR / "data/raw"

# --- モデル関連のパス ---
MODEL_OUTPUT_DIR = BASE_DIR / "models"  # 学習済みモデルの保存先
RESULT_DIR = BASE_DIR / "results"  # 可視化結果の保存先

# --- データセット設定 ---
CATEGORY = "screw"
IMAGE_SIZE = (256, 256)
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
IMAGENET_MEAN = tensor([0.485, 0.456, 0.406])
IMAGENET_STD = tensor([0.229, 0.224, 0.225])

# --- モデル設定 ---
PATCHCORE_LAYERS = ["layer2", "layer3"]

# --- 学習設定 ---
ACCELERATOR = "cpu"  # or "gpu"
DEVICES = 1
