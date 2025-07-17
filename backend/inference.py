import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import io
import logging
from pathlib import Path
import threading
import matplotlib
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from PIL import Image
import torchvision.transforms as transforms


@dataclass
class AnomalyResult:
    """単一画像の異常検出結果"""

    image_path: str
    anomaly_score: float
    confidence: float
    num_anomaly_regions: int
    max_anomaly_score: float
    heatmap_base64: str
    marking_base64: str
    processing_time: float

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "image_path": self.image_path,
            "anomaly_score": self.anomaly_score,
            "confidence": self.confidence,
            "num_anomaly_regions": self.num_anomaly_regions,
            "max_anomaly_score": self.max_anomaly_score,
            "heatmap_base64": self.heatmap_base64,
            "marking_base64": self.marking_base64,
            "processing_time": self.processing_time,
        }


@dataclass
class BatchAnomalyResult:
    """バッチ処理の結果"""

    results: List[AnomalyResult]
    total_images: int
    total_processing_time: float
    average_processing_time: float

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "results": [result.to_dict() for result in self.results],
            "total_images": self.total_images,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.average_processing_time,
        }


class PatchCoreInference:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        学習済みPatchCoreモデルを読み込む

        Args:
            model_path: 学習済みモデル(.pt)のパス
            device: 使用デバイス ('cuda' or 'cpu')
        """
        self.device = device
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.logger = logging.getLogger(__name__)

        # スレッドローカルストレージでプロットのロックを管理
        self._plot_lock = threading.Lock()

        # 画像前処理用のtransform
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        画像を前処理してテンソルに変換

        Args:
            image_path: 画像ファイルのパス

        Returns:
            前処理済みの画像テンソル
        """
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)  # type: ignore
        return image_tensor

    def image_to_base64(self, image: np.ndarray) -> str:
        """
        numpy配列をBase64エンコードされた文字列に変換

        Args:
            image: RGB画像の numpy配列

        Returns:
            Base64エンコードされた画像文字列
        """
        # PILイメージに変換
        pil_image = Image.fromarray(image.astype(np.uint8))

        # バイトストリームに変換
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)

        # Base64エンコード
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return base64_str

    def detect_anomaly_single(
        self, image_path: str, threshold: float = 0.5
    ) -> AnomalyResult:
        """
        単一画像の異常検出を実行

        Args:
            image_path: 入力画像のパス
            threshold: 異常度の閾値

        Returns:
            AnomalyResult オブジェクト
        """
        import time

        start_time = time.time()

        try:
            self.logger.info(f"Detecting anomaly for {image_path} with threshold: {threshold}")
            # 元画像を読み込み
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise ValueError(f"画像が読み込めません: {image_path}")

            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            # 前処理
            input_tensor = self.preprocess_image(image_path)

            # 推論実行
            with torch.no_grad():
                anomaly_score, anomaly_map = self.model(input_tensor)

            # テンソルをnumpy配列に変換
            anomaly_score_raw = anomaly_score.cpu().numpy()
            anomaly_map = anomaly_map.squeeze().cpu().numpy()

            # 異常度マップを0-1に正規化
            anomaly_map_normalized = (anomaly_map - anomaly_map.min()) / (
                anomaly_map.max() - anomaly_map.min() + 1e-8
            )

            # 異常スコアもマップの最大・最小値で正規化
            anomaly_score = (anomaly_score_raw - anomaly_map.min()) / (
                anomaly_map.max() - anomaly_map.min() + 1e-8
            )
            self.logger.info(f"Calculated anomaly score (raw): {anomaly_score_raw}, (normalized): {anomaly_score}")

            # ヒートマップ画像を生成（代替方法を使用）
            heatmap_image = self.create_heatmap_alternative(anomaly_map_normalized)

            # マーキング画像を生成
            marking_image = self.create_marking_image(
                original_image, anomaly_map_normalized, threshold
            )

            # 異常領域の統計情報を計算
            binary_mask = anomaly_map_normalized > threshold
            num_anomaly_regions = self.count_anomaly_regions(binary_mask)
            max_anomaly_score = float(anomaly_map_normalized.max())
            confidence = self.calculate_confidence(anomaly_map_normalized, threshold)

            # 画像をBase64に変換
            heatmap_base64 = self.image_to_base64(heatmap_image)
            marking_base64 = self.image_to_base64(marking_image)

            processing_time = time.time() - start_time

            return AnomalyResult(
                image_path=image_path,
                anomaly_score=float(anomaly_score),
                confidence=confidence,
                num_anomaly_regions=num_anomaly_regions,
                max_anomaly_score=max_anomaly_score,
                heatmap_base64=heatmap_base64,
                marking_base64=marking_base64,
                processing_time=processing_time,
            )

        except Exception as e:
            self.logger.error(f"異常検出エラー {image_path}: {str(e)}")
            raise

    def detect_anomalies_batch(
        self, image_paths: List[str], threshold: float = 0.5, max_workers: int = 4
    ) -> BatchAnomalyResult:
        """
        複数画像の異常検出を一括実行

        Args:
            image_paths: 画像パスのリスト
            threshold: 異常度の閾値
            max_workers: 並列処理のワーカー数

        Returns:
            BatchAnomalyResult オブジェクト
        """
        import time

        start_time = time.time()

        results = []

        # 並列処理で複数画像を処理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.detect_anomaly_single, image_path, threshold)
                for image_path in image_paths
            ]

            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"バッチ処理エラー: {str(e)}")
                    continue

        total_processing_time = time.time() - start_time
        average_processing_time = total_processing_time / len(results) if results else 0

        return BatchAnomalyResult(
            results=results,
            total_images=len(image_paths),
            total_processing_time=total_processing_time,
            average_processing_time=average_processing_time,
        )

    async def detect_anomalies_async(
        self, image_paths: List[str], threshold: float = 0.5, max_workers: int = 4
    ) -> BatchAnomalyResult:
        """
        非同期で複数画像の異常検出を実行

        Args:
            image_paths: 画像パスのリスト
            threshold: 異常度の閾値
            max_workers: 並列処理のワーカー数

        Returns:
            BatchAnomalyResult オブジェクト
        """
        loop = asyncio.get_event_loop()

        # CPU集約的なタスクを別スレッドで実行
        result = await loop.run_in_executor(
            None, self.detect_anomalies_batch, image_paths, threshold, max_workers
        )

        return result

    def detect_anomalies_from_directory(
        self,
        directory_path: str,
        threshold: float = 0.5,
        extensions: List[str] = [".jpg", ".jpeg", ".png", ".bmp"],
        max_workers: int = 4,
    ) -> BatchAnomalyResult:
        """
        ディレクトリ内の全画像を一括処理

        Args:
            directory_path: 画像ディレクトリのパス
            threshold: 異常度の閾値
            extensions: 処理対象の拡張子
            max_workers: 並列処理のワーカー数

        Returns:
            BatchAnomalyResult オブジェクト
        """
        directory = Path(directory_path)

        # 対象画像ファイルを取得
        image_paths = []
        for ext in extensions:
            image_paths.extend(directory.glob(f"*{ext}"))
            image_paths.extend(directory.glob(f"*{ext.upper()}"))

        image_paths = [str(path) for path in image_paths]

        if not image_paths:
            self.logger.warning(f"画像ファイルが見つかりません: {directory_path}")

        return self.detect_anomalies_batch(image_paths, threshold, max_workers)

    def create_heatmap(self, anomaly_map: np.ndarray) -> np.ndarray:  # type: ignore
        """
        異常度マップからヒートマップ画像を生成

        Args:
            anomaly_map: 正規化された異常度マップ

        Returns:
            ヒートマップ画像 (RGB)
        """
        # スレッドセーフなプロット生成
        with self._plot_lock:
            # 現在のバックエンドが'Agg'でない場合は設定
            if matplotlib.get_backend() != "Agg":
                matplotlib.use("Agg")

            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

            # ヒートマップを生成
            im = ax.imshow(anomaly_map, cmap="jet", interpolation="bilinear")

            # カラーバーを追加
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("Anomaly Score")

            ax.set_title("Anomaly Heatmap")
            ax.axis("off")

            # バッファに保存してnumpy配列に変換
            plt.tight_layout()
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # メモリを解放
            plt.close(fig)

    def create_heatmap_alternative(self, anomaly_map: np.ndarray) -> np.ndarray:  # type: ignore
        """
        Matplotlibを使わずにヒートマップ画像を生成（代替方法）

        Args:
            anomaly_map: 正規化された異常度マップ

        Returns:
            ヒートマップ画像 (RGB)
        """

        # カラーマップを手動で実装
        def apply_jet_colormap(values):
            """Jetカラーマップを適用"""
            # 値を0-255の範囲に変換
            normalized = (values * 255).astype(np.uint8)

            # Jetカラーマップの近似
            colors = np.zeros((*normalized.shape, 3), dtype=np.uint8)

            # 青から赤への変化
            colors[..., 0] = np.where(
                normalized < 128,
                0,
                np.where(normalized < 192, (normalized - 128) * 4, 255),
            )  # Red
            colors[..., 1] = np.where(
                normalized < 64,
                normalized * 4,
                np.where(normalized < 192, 255, 255 - (normalized - 192) * 4),
            )  # Green
            colors[..., 2] = np.where(
                normalized < 64,
                255,
                np.where(normalized < 128, 255 - (normalized - 64) * 4, 0),
            )  # Blue

            return colors

        # ヒートマップを生成
        height, width = anomaly_map.shape
        heatmap = apply_jet_colormap(anomaly_map)

        # サイズを調整
        if height != 224 or width != 224:
            heatmap = cv2.resize(heatmap, (224, 224))

        return heatmap

    def create_marking_image(
        self, original_image: np.ndarray, anomaly_map: np.ndarray, threshold: float
    ) -> np.ndarray:
        """
        元画像に異常部分を赤枠で囲んだマーキング画像を生成

        Args:
            original_image: 元画像 (RGB)
            anomaly_map: 正規化された異常度マップ
            threshold: 異常度の閾値

        Returns:
            マーキング画像 (RGB)
        """
        # 元画像をリサイズ
        marking_image = cv2.resize(original_image, (224, 224))

        # 異常度マップを二値化
        binary_mask = (anomaly_map > threshold).astype(np.uint8)

        # 輪郭を検出
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 異常部分を赤枠で囲む
        for contour in contours:
            if cv2.contourArea(contour) > 10:  # 小さすぎる領域は除外
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(marking_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return marking_image

    def count_anomaly_regions(self, binary_mask: np.ndarray) -> int:
        """
        異常領域の数を数える

        Args:
            binary_mask: 二値化された異常度マップ

        Returns:
            異常領域の数
        """
        binary_mask_uint8 = binary_mask.astype(np.uint8)
        contours, _ = cv2.findContours(
            binary_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 小さすぎる領域は除外
        valid_contours = [c for c in contours if cv2.contourArea(c) > 10]

        return len(valid_contours)

    def calculate_confidence(self, anomaly_map: np.ndarray, threshold: float) -> float:
        """
        検出結果の信頼度を計算

        Args:
            anomaly_map: 正規化された異常度マップ
            threshold: 異常度の閾値

        Returns:
            信頼度 (0-1)
        """
        # 異常領域の平均異常度と全体の分散から信頼度を計算
        anomaly_pixels = anomaly_map[anomaly_map > threshold]

        if len(anomaly_pixels) == 0:
            return 0.0

        # 異常領域の平均異常度
        mean_anomaly = np.mean(anomaly_pixels)

        # 全体の標準偏差
        std_all = np.std(anomaly_map)

        # 信頼度を計算 (簡易的な計算)
        confidence = min(1.0, mean_anomaly / (std_all + 1e-8))  # type: ignore

        return float(confidence)
