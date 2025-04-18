import torch

def collate_fn(batch):
    images      = torch.stack([b['images']      for b in batch])  # (B, V, 3, H, W)
    view_params = torch.stack([b['view_params'] for b in batch])  # (B, V, 2)
    meshes      = [b['mesh'] for b in batch]                     # List[Meshes]
    return {'images': images, 'view_params': view_params, 'meshes': meshes}
