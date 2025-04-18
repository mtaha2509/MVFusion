import torch
import torch.nn as nn
from transformers import AutoImageProcessor, Dinov2Model

class DinoV2Extractor(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base", device='cuda'):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model     = Dinov2Model.from_pretrained(model_name).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device

    def forward(self, images):
        # images: (B, V, 3, H, W)
        B, V, C, H, W = images.shape
        imgs = images.view(B*V, C, H, W)
        inputs = self.processor(images=imgs, return_tensors="pt")
        pix    = inputs.pixel_values.to(self.device)
        with torch.no_grad():
            out = self.model(pix)
        cls = out.last_hidden_state[:, 0, :]           # (B*V, D)
        return cls.view(B, V, -1)                     # (B, V, D)


class ViewAwareFusion(nn.Module):
    def __init__(self, dim, num_views=4):
        super().__init__()
        self.view_encoder = nn.Sequential(
            nn.Linear(dim + 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.attn = nn.MultiheadAttention(embed_dim=dim,
                                          num_heads=4,
                                          batch_first=True)

    def forward(self, feats, view_params):
        # feats: (B, V, D), view_params: (B, V, 2)
        x = torch.cat([feats, view_params], dim=-1)  # (B, V, D+2)
        x = self.view_encoder(x)                     # (B, V, D)
        out, _ = self.attn(x, x, x)                  # (B, V, D)
        return out.mean(dim=1)                       # (B, D)
