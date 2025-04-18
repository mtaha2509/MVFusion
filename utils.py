import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

def collate_fn(batch):
    images      = torch.stack([b['images']      for b in batch])  # (B, V, 3, H, W)
    view_params = torch.stack([b['view_params'] for b in batch])  # (B, V, 2)
    meshes      = [b['mesh'] for b in batch]                     # List[Meshes]
    return {'images': images, 'view_params': view_params, 'meshes': meshes}

def compute_mesh_loss(pred_mesh: Meshes, gt_mesh: Meshes, num_samples: int = 2000, device='cuda'):
    """
    Compute loss between predicted and ground truth meshes using Chamfer distance.
    Optimized for memory efficiency.
    
    Args:
        pred_mesh: Predicted mesh from SPAR3D
        gt_mesh: Ground truth mesh from ShapeNet
        num_samples: Number of points to sample from each mesh (reduced for memory)
        device: Device to compute loss on
    
    Returns:
        loss: Chamfer distance between sampled points
    """
    try:
        # Sample points from both meshes with reduced samples
        pred_points = sample_points_from_meshes(pred_mesh, num_samples=num_samples)
        gt_points = sample_points_from_meshes(gt_mesh, num_samples=num_samples)
        
        # Move points to device
        pred_points = pred_points.to(device)
        gt_points = gt_points.to(device)
        
        # Compute Chamfer distance
        loss, _ = chamfer_distance(pred_points, gt_points)
        
        # Clear memory
        del pred_points, gt_points
        torch.cuda.empty_cache()
        
        return loss
    except Exception as e:
        print(f"Error in compute_mesh_loss: {str(e)}")
        return torch.tensor(0.0, device=device)
