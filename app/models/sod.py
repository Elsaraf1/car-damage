import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# ── Architecture (must match training exactly) ───────────────────────────────

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel,
                              padding=padding * dilation,
                              dilation=dilation, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class RSU(nn.Module):
    """Residual U-block used in U2-Net lite."""
    def __init__(self, in_ch, mid_ch, out_ch, depth=4):
        super().__init__()
        self.depth = depth
        self.rebnconvin = ConvBNReLU(in_ch, out_ch)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.encoders.append(ConvBNReLU(out_ch, mid_ch))
        for i in range(1, depth):
            dil = 2 ** (i - 1) if i == depth - 1 else 1
            pad = dil
            self.encoders.append(ConvBNReLU(mid_ch, mid_ch,
                                             padding=pad, dilation=dil))

        for _ in range(depth - 1):
            self.decoders.append(ConvBNReLU(mid_ch * 2, mid_ch))

        self.out_conv = ConvBNReLU(mid_ch * 2, out_ch)

    def forward(self, x):
        residual = self.rebnconvin(x)
        enc_feats = []
        h = residual
        for enc in self.encoders:
            h = enc(h)
            enc_feats.append(h)
            if len(enc_feats) < len(self.encoders):
                h = F.max_pool2d(h, 2, stride=2, ceil_mode=True)

        h = enc_feats[-1]
        for i, dec in enumerate(reversed(self.decoders)):
            skip = enc_feats[-(i + 2)]
            h = F.interpolate(h, size=skip.shape[2:], mode='bilinear',
                              align_corners=False)
            h = dec(torch.cat([h, skip], dim=1))

        h = F.interpolate(h, size=residual.shape[2:], mode='bilinear',
                          align_corners=False)
        return self.out_conv(torch.cat([h, enc_feats[0]], dim=1)) + residual


class U2Net(nn.Module):
    """U2-Net lite (~1.1M params) — matches training config."""
    def __init__(self):
        super().__init__()
        # Encoder
        self.stage1 = RSU(3,   16, 64,  depth=7)
        self.stage2 = RSU(64,  16, 64,  depth=6)
        self.stage3 = RSU(64,  16, 64,  depth=5)
        self.stage4 = RSU(64,  16, 64,  depth=4)
        self.stage5 = RSU(64,  16, 64,  depth=4)
        # Bottleneck
        self.stage6 = RSU(64,  16, 64,  depth=4)
        # Decoder
        self.stage5d = RSU(128, 16, 64, depth=4)
        self.stage4d = RSU(128, 16, 64, depth=4)
        self.stage3d = RSU(128, 16, 64, depth=5)
        self.stage2d = RSU(128, 16, 64, depth=6)
        self.stage1d = RSU(128, 16, 64, depth=7)
        # Side output conv (1-channel)
        self.side = nn.ModuleList([
            nn.Conv2d(64, 1, 3, padding=1) for _ in range(6)
        ])
        self.fuse = nn.Conv2d(6, 1, 1)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        e1 = self.stage1(x)
        e2 = self.stage2(F.max_pool2d(e1, 2, ceil_mode=True))
        e3 = self.stage3(F.max_pool2d(e2, 2, ceil_mode=True))
        e4 = self.stage4(F.max_pool2d(e3, 2, ceil_mode=True))
        e5 = self.stage5(F.max_pool2d(e4, 2, ceil_mode=True))
        e6 = self.stage6(F.max_pool2d(e5, 2, ceil_mode=True))

        d5 = self.stage5d(torch.cat([F.interpolate(e6, size=e5.shape[2:],
                          mode='bilinear', align_corners=False), e5], dim=1))
        d4 = self.stage4d(torch.cat([F.interpolate(d5, size=e4.shape[2:],
                          mode='bilinear', align_corners=False), e4], dim=1))
        d3 = self.stage3d(torch.cat([F.interpolate(d4, size=e3.shape[2:],
                          mode='bilinear', align_corners=False), e3], dim=1))
        d2 = self.stage2d(torch.cat([F.interpolate(d3, size=e2.shape[2:],
                          mode='bilinear', align_corners=False), e2], dim=1))
        d1 = self.stage1d(torch.cat([F.interpolate(d2, size=e1.shape[2:],
                          mode='bilinear', align_corners=False), e1], dim=1))

        sides = [e1, d2, d3, d4, d5, d1]
        side_outs = []
        for i, s in enumerate(sides):
            so = self.side[i](s)
            so = F.interpolate(so, size=(H, W), mode='bilinear', align_corners=False)
            side_outs.append(so)

        fused = self.fuse(torch.cat(side_outs, dim=1))
        return [fused] + side_outs


# ── Inference ────────────────────────────────────────────────────────────────

TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def load_sod(weights_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = U2Net()
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def run_sod(model, image: Image.Image) -> Image.Image:
    orig_w, orig_h = image.size
    device = next(model.parameters()).device
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)

    # Use fused output (index 0)
    sal = torch.sigmoid(outputs[0]).squeeze().cpu().numpy()

    # Normalise to 0-255
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    sal_uint8 = (sal * 255).astype(np.uint8)

    # Apply a heatmap-style colormap (red = high saliency)
    heatmap = _apply_heatmap(sal_uint8)

    # Resize back to original and blend over original image
    heatmap_pil = Image.fromarray(heatmap).resize((orig_w, orig_h), Image.BILINEAR)
    original_rgb = image.convert("RGB")
    blended = Image.blend(original_rgb, heatmap_pil, alpha=0.55)
    return blended


def _apply_heatmap(gray: np.ndarray) -> np.ndarray:
    """Map a grayscale array to a red-yellow heatmap (no cv2 dependency)."""
    h, w = gray.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    # Red channel: always full where saliency is high
    out[..., 0] = gray
    # Green channel: ramp up at mid-high values (gives yellow tone)
    out[..., 1] = (gray.astype(np.float32) * 0.6).astype(np.uint8)
    # Blue: stays low
    out[..., 2] = 0
    return out
