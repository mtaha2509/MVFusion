import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from datasets import ShapeNetMultiView
from models import DinoV2Extractor, ViewAwareFusion
from utils import collate_fn, compute_mesh_loss
import sys
import os

# Add the SPAR3D directory to Python path
spar3d_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stable-point-aware-3d')
sys.path.append(spar3d_path)
from spar3d.system import SPAR3D
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj

import gc
import logging

def convert_spar3d_mesh_to_pytorch3d(spar3d_mesh, device='cuda'):
    """
    Convert SPAR3D mesh format to PyTorch3D format.
    
    Args:
        spar3d_mesh: SPAR3D Mesh object
        device: Device to move tensors to
    
    Returns:
        pytorch3d_mesh: PyTorch3D Meshes object
    """
    # Get vertices and faces from SPAR3D mesh
    # SPAR3D uses v_pos for vertices and t_pos_idx for faces
    vertices = spar3d_mesh.v_pos.to(device)  # (Nv, 3)
    faces = spar3d_mesh.t_pos_idx.to(device)  # (Nf, 3)
    
    # Get vertex normals if available
    if hasattr(spar3d_mesh, 'v_nrm'):
        normals = spar3d_mesh.v_nrm.to(device)  # (Nv, 3)
    else:
        normals = None
    
    # Create PyTorch3D mesh
    # Note: PyTorch3D Meshes expects lists of tensors
    pytorch3d_mesh = Meshes(
        verts=[vertices],  # List of (Nv, 3) tensors
        faces=[faces],     # List of (Nf, 3) tensors
        verts_normals=[normals] if normals is not None else None
    )
    
    return pytorch3d_mesh

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def save_checkpoint(epoch, model, optimizer, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    logging.info(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(path, model, optimizer):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']
    return 0, float('inf')

def train_fusion_model():
    device = 'cuda'
    root_dir = "/home/taha/Downloads/ShapeNetCore.v2/03001627"
    batch_size = 1  # Process one sample at a time for memory efficiency
    epochs = 50
    lr = 1e-4
    
    # Create checkpoint directory
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Memory management
    torch.cuda.empty_cache()
    gc.collect()

    # Instantiate models
    dino = DinoV2Extractor(device=device)
    fusion = ViewAwareFusion(dim=768).to(device)
    
    # Initialize SPAR3D with low VRAM mode
    try:
        spar3d = SPAR3D.from_pretrained(
            "stabilityai/stable-point-aware-3d",
            config_name="config.yaml",
            weight_name="model.safetensors",
            low_vram_mode=True
        ).to(device)
        spar3d.eval()
    except Exception as e:
        logging.error(f"Failed to initialize SPAR3D: {str(e)}")
        raise
    
    opt = torch.optim.Adam(fusion.parameters(), lr=lr)
    
    # Load checkpoint if exists
    start_epoch, best_loss = load_checkpoint(
        os.path.join(checkpoint_dir, 'latest.pt'),
        fusion,
        opt
    )

    # Dataset & DataLoader with reduced workers
    dataset = ShapeNetMultiView(root_dir, device=device)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=False,
        collate_fn=collate_fn
    )

    for ep in range(start_epoch + 1, epochs + 1):
        total_loss = 0.0
        successful_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Clear memory before processing batch
                torch.cuda.empty_cache()
                gc.collect()

                # Move data to GPU
                imgs = batch['images'].to(device)
                vpars = batch['view_params'].to(device)
                gt_meshes = batch['meshes']

                # Get DINOv2 features for each view
                with torch.no_grad():
                    feats = dino(imgs)  # (B, V, D)

                # Fuse features while preserving distribution
                fused = fusion(feats, vpars)  # (B, D)

                # Generate 3D mesh using SPAR3D
                with torch.no_grad():
                    try:
                        # SPAR3D expects DINOv2 features, which our fused features should match
                        spar3d_mesh, _ = spar3d.run_image(
                            fused.unsqueeze(0),  # Add batch dimension
                            bake_resolution=512,
                            remesh="none"
                        )
                        
                        # Convert SPAR3D mesh to PyTorch3D format
                        pred_mesh = convert_spar3d_mesh_to_pytorch3d(spar3d_mesh, device)
                        
                        # Compute loss between predicted and ground truth meshes
                        loss = compute_mesh_loss(pred_mesh, gt_meshes[0], num_samples=1000, device=device)
                        
                        # Backward pass
                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                        total_loss += loss.item()
                        successful_batches += 1
                        
                        # Clear memory
                        del spar3d_mesh, pred_mesh
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        logging.warning(f"Error in SPAR3D processing: {str(e)}")
                        continue

                # Print progress
                if (batch_idx + 1) % 10 == 0:
                    logging.info(f"Epoch {ep}/{epochs}  Batch {batch_idx+1}/{len(dataloader)}  Loss: {loss.item():.4f}")
                    
                # Save checkpoint every 100 batches
                if (batch_idx + 1) % 100 == 0:
                    save_checkpoint(
                        ep,
                        fusion,
                        opt,
                        loss.item(),
                        os.path.join(checkpoint_dir, f'checkpoint_epoch{ep}_batch{batch_idx}.pt')
                    )
                    
            except Exception as e:
                logging.error(f"Error in batch {batch_idx}: {str(e)}")
                continue

        if successful_batches > 0:
            avg_loss = total_loss / successful_batches
            logging.info(f"Epoch {ep}/{epochs}  Average Loss: {avg_loss:.4f}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(
                    ep,
                    fusion,
                    opt,
                    avg_loss,
                    os.path.join(checkpoint_dir, 'best.pt')
                )
            
            # Save latest checkpoint
            save_checkpoint(
                ep,
                fusion,
                opt,
                avg_loss,
                os.path.join(checkpoint_dir, 'latest.pt')
            )

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    train_fusion_model()
