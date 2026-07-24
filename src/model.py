"""
ConViTX reproduction (PyTorch).

Reference: Thakur, Chaturvedi, Seal, Khanna, Sheorey, Ojha, "An Ultra Lightweight
Interpretable Convolution-Vision Transformer Fusion Model for Plant Disease
Identification: ConViTX", IEEE TCBB, 2025. (official code:
https://github.com/Image-and-Vision-Engineering-Group/ConViTX -- not consulted;
this is an independent clean-room reimplementation from the paper text.)

Architecture per Section II-B / Fig. 2 / Algorithm 1:
  - CNN branch: first two blocks of pretrained MobileNetV2 -> SE block
  - ViT branch: 7x7 patches -> linear projection -> positional embedding ->
    8 transformer encoder blocks -> reshape to spatial map -> 3x 3x3 depthwise
    separable conv layers -> SE block
  - Fusion: concat(CNN, ViT) -> MobileNetV2-style block (16 filters, 3x3, s=2)
    -> SE block -> GAP -> Linear classifier

Known paper ambiguities/contradictions resolved here (documented, not silently
guessed away -- see ConViTX_repro/README.md "Known deviations"):
  1. MHA head count: Section III-C ablation text says 16 heads performed best
     (Table II(a)), but the paragraph immediately preceding Section III-D
     ("the selected ConViTX architecture... each layer has 4 MHA heads")
     states 4. We use 4, matching the final-architecture paragraph, since
     that is the paper's own stated configuration for the deployed model.
  2. Spatial resolution matching between the CNN branch (56x56, from
     MobileNetV2 blocks 1-2 on a 224x224 input) and the ViT branch (32x32,
     from 7x7 patches on the same input) is not specified beyond "the Conv
     block... helps match the output dimension of the ViT module with that
     of the CNN module". We bilinearly upsample the ViT branch's spatial map
     32x32 -> 56x56 after the depthwise conv stage, and use a 1x1 pointwise
     conv to match channel count (24, mirroring the CNN branch), before SE
     and concatenation.
  3. "Projection dimension" is stated as 48 in the final-architecture
     paragraph but Table II(c)'s best AC (99.54) is actually listed under
     the "64" row -- we use 48, per the same reasoning as (1).
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import MobileNet_V2_Weights
from torchvision.models.mobilenetv2 import InvertedResidual


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block (channel attention)."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)
        s = self.fc(s).view(b, c, 1, 1)
        return x * s


class CNNBranch(nn.Module):
    """First two MobileNetV2 blocks (stem + stage1[c=16] + stage2[c=24]) + SE."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.mobilenet_v2(weights=weights)
        # features[0]=stem conv, [1]=block1 (t=1,c=16,s=1), [2:4]=block2 (t=6,c=24,s=2,s=1)
        self.stem_and_blocks = backbone.features[0:4]
        self.out_channels = 24
        self.se = SEBlock(self.out_channels)

    def forward(self, x):
        x = self.stem_and_blocks(x)  # (B, 24, 56, 56) for 224x224 input
        return self.se(x)


class TransformerEncoderBlock(nn.Module):
    """Standard pre-norm ViT encoder block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.bn(x))


class ViTBranch(nn.Module):
    """7x7-patch ViT over the image, reshaped back to a spatial feature map."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 7,
        embed_dim: int = 48,
        depth: int = 8,
        num_heads: int = 4,
        out_channels: int = 24,
        out_size: int = 56,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.grid = img_size // patch_size  # 32
        self.num_patches = self.grid ** 2
        self.embed_dim = embed_dim
        self.out_size = out_size

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(embed_dim, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.dsc_layers = nn.Sequential(
            DepthwiseSeparableConv(embed_dim, embed_dim),
            DepthwiseSeparableConv(embed_dim, embed_dim),
            DepthwiseSeparableConv(embed_dim, out_channels),
        )
        self.se = SEBlock(out_channels)

    def forward(self, x):
        b = x.shape[0]
        tokens = self.patch_embed(x)  # (B, embed_dim, grid, grid)
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        tokens = tokens + self.pos_embed
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        fmap = tokens.transpose(1, 2).reshape(b, self.embed_dim, self.grid, self.grid)
        fmap = self.dsc_layers(fmap)
        fmap = nn.functional.interpolate(
            fmap, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False
        )
        return self.se(fmap)


class FusionModule(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        # "MobileNetV2 block" with 16 filters, 3x3, stride=2
        self.mbv2_block = InvertedResidual(in_channels, 16, stride=2, expand_ratio=6)
        self.se = SEBlock(16)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.mbv2_block(x)
        x = self.se(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


class ConViTX(nn.Module):
    def __init__(self, num_classes: int, pretrained_cnn: bool = True):
        super().__init__()
        self.cnn_branch = CNNBranch(pretrained=pretrained_cnn)
        self.vit_branch = ViTBranch(
            out_channels=self.cnn_branch.out_channels, out_size=56
        )
        fused_channels = self.cnn_branch.out_channels + self.vit_branch.dsc_layers[-1].pointwise.out_channels
        self.fusion = FusionModule(fused_channels, num_classes)

    def forward(self, x):
        f_cnn = self.cnn_branch(x)
        f_vit = self.vit_branch(x)
        fused = torch.cat([f_cnn, f_vit], dim=1)
        return self.fusion(fused)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = ConViTX(num_classes=38)
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    print("output shape:", y.shape)
    print("trainable params:", count_params(m))
