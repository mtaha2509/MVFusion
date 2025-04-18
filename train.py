import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from datasets import ShapeNetMultiView
from models   import DinoV2Extractor, ViewAwareFusion
from utils    import collate_fn

def train_fusion_model():
    device     = 'cuda'
    root_dir   = "/home/taha/Downloads/ShapeNetCore.v2/03001627"
    batch_size = 4
    epochs     = 50
    lr         = 1e-4

    # Instantiate models
    dino   = DinoV2Extractor(device=device)
    fusion = ViewAwareFusion(dim=768).to(device)
    opt    = torch.optim.Adam(fusion.parameters(), lr=lr)

    # Dataset & DataLoader
    dataset    = ShapeNetMultiView(root_dir, device=device)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,          # or 0 to avoid multiprocessing altogether
        pin_memory=False,       # disable pinning to avoid GPU-tensor pin errors :contentReference[oaicite:3]{index=3}
        collate_fn=collate_fn
    )

    for ep in range(1, epochs+1):
        total_loss = 0.0
        for batch in dataloader:
            # Move only here to GPU
            imgs   = batch['images'].to(device)
            vpars  = batch['view_params'].to(device)
            # meshes remains a CPU list or you can move them as needed

            # DINO features (no grad)
            with torch.no_grad():
                feats = dino(imgs)

            # Fuse
            fused = fusion(feats, vpars)

            # Placeholder loss
            loss = torch.tensor(0.0, device=device)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        print(f"Epoch {ep}/{epochs}  Loss: {total_loss/len(dataloader):.4f}")


if __name__ == "__main__":
    # Must set spawn _before_ starting workers
    mp.set_start_method('spawn', force=True)
    train_fusion_model()
