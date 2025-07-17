from pathlib import Path
from typing import Tuple
import timm
import numpy as np
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn.functional as F

from utils import NativeGaussianBlur, get_coreset_idx_randomp, get_tqdm_params

EXPORT_DIR = Path("./exports")
EXPORT_DIR.mkdir(exist_ok=True)


class KNNExtractor(torch.nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        out_indices: Tuple = None,  # type: ignore
        pool_last: bool = False,
    ):
        super().__init__()
        self.feature_extractor = timm.create_model(
            backbone_name,
            out_indices=out_indices,
            features_only=True,
            pretrained=True,
            exportable=True,
        )
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

        self.pool = torch.nn.AdaptiveAvgPool2d(1) if pool_last else None
        self.backbone_name = backbone_name
        self.out_indices = out_indices
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = self.feature_extractor.to(self.device)

    def extract(self, x: Tensor):
        with torch.no_grad():
            feature_maps = self.feature_extractor(x.to(self.device))
        feature_maps = [fmap.to("cpu") for fmap in feature_maps]
        if self.pool is not None:
            return feature_maps[:-1], self.pool(feature_maps[-1])
        else:
            return feature_maps

    def fit(self, _: DataLoader):
        raise NotImplementedError

    def predict(self, _: Tensor):
        raise NotImplementedError

    def evaluate(self, test_dl: DataLoader) -> Tuple[float, float]:
        image_preds = []
        image_labels = []
        pixel_preds = []
        pixel_labels = []

        for sample, mask, label in tqdm(test_dl, **get_tqdm_params()):
            z_score, fmap = self.forward(sample)
            image_preds.append(z_score.numpy())
            image_labels.append(label)
            pixel_preds.extend(fmap.flatten().numpy())
            pixel_labels.extend(mask.flatten().numpy())

        image_labels = np.stack(image_labels)
        image_preds = np.stack(image_preds)

        image_rocauc = roc_auc_score(image_labels, image_preds)
        pixel_rocauc = roc_auc_score(pixel_labels, pixel_preds)

        return image_rocauc, pixel_rocauc  # type: ignore

    def get_parameters(self, extra_params: dict = None) -> dict:  # type: ignore
        return {
            "backbone_name": self.backbone_name,
            "out_indices": self.out_indices,
            **extra_params,
        }


class PatchCore(KNNExtractor):
    def __init__(
        self,
        f_coreset: float = 0.01,
        backbone_name: str = "resnet18",
        coreset_eps: float = 0.90,
    ):
        super().__init__(backbone_name=backbone_name, out_indices=(2, 3))
        self.f_coreset = f_coreset
        self.coreset_eps = coreset_eps
        self.image_size = 224
        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.blur = NativeGaussianBlur()
        self.patch_lib = []
        self.resize = None

    def fit(self, train_dl):  # type: ignore
        for sample, _ in tqdm(train_dl, **get_tqdm_params()):
            feature_maps = self.extract(sample)
            if self.resize is None:
                self.largest_fmap_size = feature_maps[0].shape[-2:]  # type: ignore
                self.resize = torch.nn.AdaptiveAvgPool2d(self.largest_fmap_size)

            resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]
            patch = torch.cat(resized_maps, 1)
            patch = patch.reshape(patch.shape[1], -1).T
            patch = F.normalize(patch, p=2.0, dim=1)
            self.patch_lib.append(patch)

        self.patch_lib = torch.cat(self.patch_lib, 0)

        if self.f_coreset < 1:
            self.coreset_idx = get_coreset_idx_randomp(
                self.patch_lib,
                n=int(self.f_coreset * self.patch_lib.shape[0]),
                eps=self.coreset_eps,
            )
            self.patch_lib = self.patch_lib[self.coreset_idx]

    def forward(self, sample):
        # 特徴マップ抽出
        feature_maps = self.extract(sample)

        # 特徴マップのリサイズと正規化
        resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]  # type: ignore
        patch = torch.cat(resized_maps, 1)  # [B, C, H, W] → [B, C, H, W]
        patch = patch.reshape(patch.shape[1], -1).T  # [N_patches, C]
        patch = F.normalize(patch, p=2.0, dim=1)  # L2正規化

        # すべてのパッチとの距離を計算
        dist = torch.cdist(patch, self.patch_lib)  # type: ignore
        min_val, min_idx = torch.min(dist, dim=1)  # [N_patches]

        # 最も異常なパッチ (最大の min_val を持つパッチ)
        s_star, s_idx = torch.max(min_val, dim=0)

        # m_test_star = テスト画像内の最も異常なパッチ
        m_test_star = patch[s_idx].unsqueeze(0)  # [1, C]

        # m_star = 訓練時コアセット内で最も近いパッチ
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)  # [1, C]

        # m_star に対する近傍パッチを取得
        k = 5
        dists_m_star = torch.cdist(m_star, self.patch_lib)  # type: ignore
        _, nn_idx = torch.topk(dists_m_star, k=k, largest=False)

        nb_vectors = self.patch_lib[nn_idx[0]]  # [k, C]
        nb_dists = torch.linalg.norm(m_test_star - nb_vectors, dim=1)  # [k]

        # 論文式 (7) の重み計算
        numerator = torch.exp(torch.norm(m_test_star - m_star))
        denominator = torch.sum(torch.exp(nb_dists))
        w = 1.0 - (numerator / denominator)

        # 最終異常スコア s
        s = w * s_star

        # 異常マップ生成
        s_map = min_val.view(1, 1, *self.largest_fmap_size)
        s_map = F.interpolate(
            s_map, size=(self.image_size, self.image_size), mode="bilinear"
        )
        s_map = self.blur(s_map)

        return s, s_map

    def get_parameters(self):  # type: ignore
        return super().get_parameters(
            {
                "f_coreset": self.f_coreset,
            }
        )

    def export(self, save_name: str):
        scripted_predictor = torch.jit.script(self)
        scripted_predictor.save(f"{EXPORT_DIR}/{save_name}.pt")
