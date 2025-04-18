import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.dim = dim
        self.num_views = num_views
        
        # View parameter encoding with positional information
        self.view_encoder = nn.Sequential(
            nn.Linear(2, dim // 4),  # Reduced dimension for view encoding
            nn.LayerNorm(dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
            nn.LayerNorm(dim)
        )
        
        # Cross-view attention for feature refinement
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=8,  # Increased heads for better view interaction
            batch_first=True
        )
        
        # Feature refinement with residual connections
        self.feature_refinement = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
        
        # View importance prediction
        self.view_importance = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, feats, view_params):
        """
        Args:
            feats: (B, V, D) - DINOv2 features
            view_params: (B, V, 2) - view parameters
        """
        B, V, D = feats.shape
        
        # 1. View-aware feature enhancement
        # Encode view parameters
        view_emb = self.view_encoder(view_params)  # (B, V, D)
        
        # Add view embeddings to features with residual connection
        x = self.norm1(feats + view_emb)
        
        # 2. Cross-view attention for feature refinement
        # Compute attention between views
        attn_out, attn_weights = self.cross_attn(x, x, x)  # (B, V, D)
        x = self.norm2(x + attn_out)
        
        # 3. Feature refinement
        refined = self.feature_refinement(x)  # (B, V, D)
        x = self.norm3(x + refined)
        
        # 4. View importance weighting
        # Predict importance of each view
        view_weights = self.view_importance(x)  # (B, V, 1)
        
        # Normalize weights
        view_weights = F.softmax(view_weights, dim=1)
        
        # Weighted average of features
        fused = (x * view_weights).sum(dim=1)  # (B, D)
        
        return fused
