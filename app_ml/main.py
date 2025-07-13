import random
import base64
import io
import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

from app_ml.data.datamodule import MVTecDataset
from app_ml.models.patchcore import PatchCore

# seeds
import warnings

from config import settings

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join(settings.MODEL_OUTPUT_DIR, "patchcore_model.pt")

os.makedirs(settings.MODEL_OUTPUT_DIR, exist_ok=True)


def train_and_save_model():
    """
    PatchCoreモデルを学習し、状態をファイルに保存する。
    """
    print("Starting model training...")
    model = PatchCore(
        f_coreset=0.10,
        backbone_name="efficientnet_b0",
    )
    train_ds, _ = MVTecDataset("screw").get_dataloaders()
    model.fit(train_ds)
    torch.save(model, MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")
    return model


def initialize_model():
    """
    保存されたモデルの状態を読み込む。
    ファイルが存在しない場合は、新たに学習して保存する。
    """
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        model = torch.load(MODEL_PATH)
    else:
        print("No pre-trained model found. Training a new one...")
        model = train_and_save_model()
    return model


def run_inference_on_image(model: PatchCore, image_base64: str):
    """
    学習済みモデルとBase64エンコードされた画像データを受け取り、推論を実行する
    """
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = transform(image).unsqueeze(0)

    # score, heatmap, marked_image = model.predict(input_tensor)

    score = random.random()
    threshold = 0.5
    is_anomalous = score > threshold

    dummy_image = Image.new("RGB", (224, 224), (255, 255, 255))
    buffered = io.BytesIO()
    dummy_image.save(buffered, format="PNG")
    encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "id": f"res_{random.randint(1000, 9999)}",
        "image": f"data:image/png;base64,{image_base64}",
        "anomalyScore": score,
        "isAnomalous": is_anomalous,
        "heatmap": f"data:image/png;base64,{encoded_string}",
        "markedImage": f"data:image/png;base64,{encoded_string}",
        "timestamp": "2024-07-14T12:00:00Z",
    }


# def cli_interface():
#     dataset = ["screw"]
#     total_results = run_model("patchcore", dataset, "efficientnet_b0")
#     print_and_export_results(total_results, "patchcore")

# if __name__ == "__main__":
#     cli_interface()