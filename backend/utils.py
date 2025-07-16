import sys
from torch import nn, tensor
from torch.nn import functional as F
import torch
from sklearn import random_projection
from tqdm import tqdm


TQDM_PARAMS = {
    "file": sys.stdout,
    "bar_format": "   {l_bar}{bar:10}{r_bar}{bar:-10b}",
}


def get_tqdm_params():
    return TQDM_PARAMS


class NativeGaussianBlur(nn.Module):
    def __init__(self, channels: int = 1, kernel_size: int = 21, sigma: float = 4.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2
        self.register_buffer("kernel", self.create_gaussian_kernel())

    def create_gaussian_kernel(self):
        coords = torch.arange(self.kernel_size).float() - self.kernel_size // 2
        x = coords.repeat(self.kernel_size, 1)
        y = x.t()
        gaussian = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * self.sigma**2))
        kernel = gaussian / gaussian.sum()
        return kernel.view(1, 1, self.kernel_size, self.kernel_size).repeat(
            self.channels, 1, 1, 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel, padding=self.padding, groups=self.channels)  # type: ignore


def get_coreset_idx_randomp(
    z_lib: tensor,  # type: ignore
    n: int = 1000,
    eps: float = 0.90,
    float16: bool = True,
    force_cpu: bool = False,
) -> tensor:  # type: ignore
    print(f"   Fitting random projections. Start dim = {z_lib.shape}.")
    try:
        transformer = random_projection.SparseRandomProjection(eps=eps)
        z_lib = torch.tensor(transformer.fit_transform(z_lib))
        print(f"   DONE.                 Transformed dim = {z_lib.shape}.")
    except ValueError:
        print("   Error: could not project vectors. Please increase `eps`.")

    select_idx = 0
    last_item = z_lib[select_idx : select_idx + 1]
    coreset_idx = [torch.tensor(select_idx)]
    min_distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)

    if float16:
        last_item = last_item.half()
        z_lib = z_lib.half()
        min_distances = min_distances.half()
    if torch.cuda.is_available() and not force_cpu:
        last_item = last_item.to("cuda")
        z_lib = z_lib.to("cuda")
        min_distances = min_distances.to("cuda")

    for _ in tqdm(range(n - 1), **TQDM_PARAMS):
        distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)
        min_distances = torch.minimum(distances, min_distances)
        select_idx = torch.argmax(min_distances)

        last_item = z_lib[select_idx : select_idx + 1]
        min_distances[select_idx] = 0
        coreset_idx.append(select_idx.to("cpu"))

    return torch.stack(coreset_idx)
