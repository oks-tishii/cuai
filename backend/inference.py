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
                transforms.Resize(256),
                transforms.CenterCrop(224),
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
        self, image_path: str, threshold: float = 0.3
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
            self.logger.info(
                f"Detecting anomaly for {image_path} with threshold: {threshold}"
            )
            # 元画像を読み込み
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise ValueError(f"画像が読み込めません: {image_path}")

            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            # 前処理
            input_tensor = self.preprocess_image(image_path)

            # 推論実行
            # with torch.no_grad():
            #     anomaly_score, anomaly_map = self.model(input_tensor)
            with torch.no_grad():
                anomaly_score_tensor, anomaly_map_tensor = self.model(input_tensor)

            # テンソルをnumpy配列に変換（マップ：(B,1,H,W) -> (H,W)）
            anomaly_score_raw = float(anomaly_score_tensor.detach().cpu().item())
            anomaly_map_np = anomaly_map_tensor.squeeze().numpy()

            # 異常度マップを0-1に正規化
            am_min = float(anomaly_map_tensor.min())
            am_max = float(anomaly_map_tensor.max())
            anomaly_map_vis = (anomaly_map_np - am_min) / (am_max - am_min + 1e-8)

            # 異常スコアもマップの最大・最小値で正規化
            anomaly_score_normalized = float(anomaly_score_raw)

            print(
                f"Calculated anomaly score (raw): {anomaly_score_raw}, (normalized): {anomaly_score_tensor}"
            )

            # ヒートマップ画像を生成（代替方法を使用）
            heatmap_image = self.create_heatmap_alternative(anomaly_map_vis)  # type: ignore

            # マーキング画像を生成
            marking_image = self.create_marking_image(
                original_image, heatmap_image, threshold
            )

            # 異常領域の統計情報を計算
            binary_mask = anomaly_map_vis > threshold
            num_anomaly_regions = self.count_anomaly_regions(binary_mask)
            max_anomaly_score = float(anomaly_map_vis.max())
            confidence = self.calculate_confidence(anomaly_map_vis, threshold)

            # 画像をBase64に変換
            heatmap_base64 = self.image_to_base64(heatmap_image)
            marking_base64 = self.image_to_base64(marking_image)

            processing_time = time.time() - start_time

            return AnomalyResult(
                image_path=image_path,
                anomaly_score=float(anomaly_score_normalized),
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

    # def create_heatmap_alternative(self, anomaly_map: np.ndarray) -> np.ndarray:  # type: ignore
    #     """
    #     Matplotlibを使わずにヒートマップ画像を生成（代替方法）

    #     Args:
    #         anomaly_map: 正規化された異常度マップ

    #     Returns:
    #         ヒートマップ画像 (RGB)
    #     """

    #     # カラーマップを手動で実装
    #     def apply_jet_colormap(values):
    #         """Jetカラーマップを適用"""
    #         # 値を0-255の範囲に変換
    #         normalized = (values * 255).astype(np.uint8)

    #         # Jetカラーマップの近似
    #         colors = np.zeros((*normalized.shape, 3), dtype=np.uint8)

    #         # 青から赤への変化
    #         colors[..., 0] = np.where(
    #             normalized < 128,
    #             0,
    #             np.where(normalized < 192, (normalized - 128) * 4, 255),
    #         )  # Red
    #         colors[..., 1] = np.where(
    #             normalized < 64,
    #             normalized * 4,
    #             np.where(normalized < 192, 255, 255 - (normalized - 192) * 4),
    #         )  # Green
    #         colors[..., 2] = np.where(
    #             normalized < 64,
    #             255,
    #             np.where(normalized < 128, 255 - (normalized - 64) * 4, 0),
    #         )  # Blue

    #         return colors

    #     # ヒートマップを生成
    #     height, width = anomaly_map.shape
    #     heatmap = apply_jet_colormap(anomaly_map)

    #     # サイズを調整
    #     if height != 224 or width != 224:
    #         heatmap = cv2.resize(heatmap, (224, 224))

    #     return heatmap

    def create_heatmap_alternative(
        self,
        anomaly_map: np.ndarray,  # type: ignore
        *,
        out_size: int | tuple[int, int] = 224,
        colormap: str = "jet",
        auto_normalize: bool = True,
        clip: tuple[float, float] | None = None,
        to_rgba: bool = False,
        overlay_bgr: np.ndarray | None = None,
        overlay_alpha: float = 0.5,
    ) -> np.ndarray:
        if anomaly_map.ndim != 2:
            raise ValueError(f"anomaly_map must be 2D, got shape={anomaly_map.shape}")
        if np.isnan(anomaly_map).any():
            raise ValueError("anomaly_map contains NaN.")
        amap = anomaly_map.astype(np.float32)

        # クリップ
        if clip is not None:
            lo, hi = clip
            amap = np.clip(amap, lo, hi)

        # 自動正規化（全要素一定値ならゼロマップに）
        if auto_normalize:
            amin, amax = float(amap.min()), float(amap.max())
            if amax - amin > 1e-12:
                amap = (amap - amin) / (amax - amin)
            else:
                amap = np.zeros_like(amap)

        # 出力サイズ
        if isinstance(out_size, int):
            target_w = target_h = out_size
        else:
            target_w, target_h = out_size
        if amap.shape[0] != target_h or amap.shape[1] != target_w:
            # OpenCV は (W,H) ではなく (width,height)
            amap_resized = cv2.resize(
                amap, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )
        else:
            amap_resized = amap

        # --- Colormap 実装 ---
        def apply_colormap(v01: np.ndarray, name: str) -> np.ndarray:
            v = np.clip(v01, 0.0, 1.0)
            if name == "jet":
                r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
                g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
                b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
                rgb = np.stack([r, g, b], axis=-1)
            elif name == "magma":
                c_lin = np.linspace(0, 1, 256, dtype=np.float32)

                # 近似パレット（粗い定義）：R,G,B の各曲線
                R = np.clip(1.8 * c_lin**1.2 - 0.1, 0, 1)
                G = np.clip((c_lin**1.5) * 1.3, 0, 1)
                B = np.clip(0.8 * (c_lin**0.7), 0, 1)
                lut = np.stack([R, G, B], axis=1)
                idx = (v * 255).astype(np.int32)
                rgb = lut[idx]
            elif name == "turbo":
                r = 0.135 + 4.0 * v - 4.5 * v**2
                g = -0.25 + 2.8 * v - 1.8 * v**2
                b = 0.45 - 1.5 * v + 2.0 * v**2
                rgb = np.stack([r, g, b], axis=-1)
                rgb = np.clip(rgb, 0, 1)
            else:
                raise ValueError(f"Unsupported colormap: {name}")
            return (rgb * 255).astype(np.uint8)

        heat_rgb = apply_colormap(amap_resized, colormap)

        # RGBA 変換
        if to_rgba:
            alpha_channel = np.full(
                (heat_rgb.shape[0], heat_rgb.shape[1], 1), 255, dtype=np.uint8
            )
            heat_rgba = np.concatenate([heat_rgb, alpha_channel], axis=-1)
            result = heat_rgba
        else:
            result = heat_rgb

        # オーバーレイ（元画像 BGR→RGB 換算 & alpha blend）
        if overlay_bgr is not None:
            if overlay_bgr.shape[:2] != (target_h, target_w):
                overlay_bgr = cv2.resize(
                    overlay_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA
                )
            base_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            # ヒートマップは 0~255
            blended = (1 - overlay_alpha) * base_rgb.astype(
                np.float32
            ) + overlay_alpha * heat_rgb.astype(np.float32)
            result = blended.clip(0, 255).astype(np.uint8)
            if to_rgba:
                alpha_channel = np.full((target_h, target_w, 1), 255, dtype=np.uint8)
                result = np.concatenate([result, alpha_channel], axis=-1)

        return result

    def create_marking_image(
        self,
        original_image: np.ndarray,
        anomaly_map: np.ndarray,
        threshold: float,
        overlay_heatmap: bool = True,
        alpha: float = 0.5,
    ) -> np.ndarray:
        if isinstance(anomaly_map, np.ndarray):
            heatmap_pil = Image.fromarray(anomaly_map)
        else:
            heatmap_pil = anomaly_map

        original_pil = Image.fromarray(original_image)

        if heatmap_pil.size != original_pil.size:
            base_pil = original_pil.resize(heatmap_pil.size)
        else:
            base_pil = original_pil

        if overlay_heatmap:
            base_np = np.asarray(base_pil).astype(np.float32) / 255.0
            heat_np = np.asarray(heatmap_pil).astype(np.float32) / 255.0

            overlay_np = (1 - alpha) * base_np + alpha * heat_np

            overlay_np = (overlay_np * 255).clip(0, 255).astype(np.uint8)
            return overlay_np
        else:
            return np.asarray(base_pil)

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
