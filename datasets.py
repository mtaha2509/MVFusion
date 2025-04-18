import os
import torch
from torch.utils.data import Dataset
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, MeshRenderer, MeshRasterizer, HardPhongShader
)
from pytorch3d.renderer.mesh.textures import TexturesVertex

class ShapeNetMultiView(Dataset):
    def __init__(self,
                 root_dir,
                 views=[(0,0),(180,0),(90,0),(270,0)],
                 image_size=224,
                 device='cuda'):
        self.device = device
        self.views = views
        self.image_size = image_size

        # Gather all (model_id, .obj) pairs
        self.models = []
        for model_id in os.listdir(root_dir):
            mesh_p = os.path.join(root_dir, model_id, 'models', 'model_normalized.obj')
            if os.path.isfile(mesh_p):
                self.models.append((model_id, mesh_p))
        if len(self.models) == 0:
            raise RuntimeError(f"No meshes found under {root_dir}")

        # Setup renderer once
        raster_settings = RasterizationSettings(
            image_size=image_size, blur_radius=0.0, faces_per_pixel=1
        )
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings),
            shader=HardPhongShader(device=device)
        )

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        model_id, mesh_path = self.models[idx]
        # Load .obj (and any UV textures it references)
        mesh = load_objs_as_meshes([mesh_path], device=self.device, load_textures=True)

        # If there is no <model-id>/images/ folder with real jpg/png, assign white vertex colors
        tex_dir = os.path.join(os.path.dirname(os.path.dirname(mesh_path)), 'images')
        if not (os.path.isdir(tex_dir) and any(f.lower().endswith(('.jpg','.png'))
                                              for f in os.listdir(tex_dir))):
            verts = mesh.verts_packed()
            white = torch.ones((1, verts.size(0), 3), device=self.device)
            mesh.textures = TexturesVertex(verts_features=white)

        # Render each view
        images, view_params = [], []
        for azim, elev in self.views:
            R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
            cams = FoVPerspectiveCameras(device=self.device, R=R, T=T)
            rend = self.renderer(mesh, cameras=cams)     # (1, H, W, 3)
            img = rend[..., :3].permute(0,3,1,2).squeeze(0) / 255.0
            images.append(img)
            view_params.append([azim, elev])

        return {
            'images': torch.stack(images),                # (V,3,H,W)
            'view_params': torch.tensor(view_params,      # (V,2)
                                       device=self.device),
            'mesh': mesh                                 # a PyTorch3D Meshes object
        }
