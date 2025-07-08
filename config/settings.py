import os
import torch
from typing import Dict, Any

# プロジェクトのルートディレクトリパスを取得
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 全体設定 (CONFIG辞書)
# ここにすべてのハイパーパラメータやパスを一元管理します。
CONFIG: Dict[str, Any] = {
    "image_size": 128,  # 画像のリサイズサイズ (VAEモデルの入力サイズに合わせる)
    "z_dim": 100,  # VAEの潜在空間の次元数
    "input_channels": 3,  # 入力画像のチャンネル数 (RGB画像の場合は3)
    "batch_size": 32,  # 学習時のバッチサイズ
    "epochs": 1000,  # 学習エポック数
    "learning_rate": 2e-4,  # Adamオプティマイザの学習率
    "betas": (0.5, 0.999),  # Adamオプティマイザのbetaパラメータ
    "weight_decay": 1e-5,  # L2正則化の強度
    "log_interval_batch": 50,  # 学習ログを出力するバッチの頻度
    "save_interval_epoch": 100,  # モデルを保存するエポックの頻度
    # 生の学習画像データが格納されているディレクトリ
    "train_image_dir": os.path.join(ROOT_DIR, "data", "raw", "transister", "good"),
    # 学習済みモデルのチェックポイントを保存するディレクトリ
    "model_save_path": os.path.join(ROOT_DIR, "models"),
    # 推論時にロードするモデルのパス (例: 'anomaly_det_model_best.pt' など)
    # 'inference_model_path': os.path.join(ROOT_DIR, 'models', 'anomaly_det_model_best.pt'),
}

# デバイス設定: GPU (CUDA) が利用可能であればGPUを使用し、そうでなければCPUを使用
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"使用デバイス: {DEVICE}")
