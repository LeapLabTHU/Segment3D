import hydra
import torch

from models.mask3d import Mask3D
from models.mask3d_no_aux import Mask3D_no_aux
from utils.utils import (
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)

from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.experimental import initialize, compose

import albumentations as A
import MinkowskiEngine as ME
import numpy as np
import open3d as o3d


class InstanceSegmentation(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = hydra.utils.instantiate(cfg.model)


    def forward(self, x, point2segment=None, raw_coordinates=None):
        return self.model(x, point2segment=point2segment, raw_coordinates=raw_coordinates)
    

def get_model(cfg):
  
    model = InstanceSegmentation(cfg)

    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(
            cfg, model
        )
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    return model


def load_mesh(pcl_file):
    
    # load point cloud
    input_mesh_path = pcl_file
    mesh = o3d.io.read_triangle_mesh(input_mesh_path)
    return mesh

def prepare_data(cfg, mesh, point2segment, device):
    
    # normalization for point cloud features
    color_mean = (0.47793125906962, 0.4303257521323044, 0.3749598901421883)
    color_std = (0.2834475483823543, 0.27566157565723015, 0.27018971370874995)
    normalize_color = A.Normalize(mean=color_mean, std=color_std)
    
    points = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    colors = colors * 255.

    pseudo_image = colors.astype(np.uint8)[np.newaxis, :, :]
    colors = np.squeeze(normalize_color(image=pseudo_image)["image"])

    coords = np.floor(points / cfg.data.voxel_size)
    _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
        coordinates=coords,
        features=colors,
        return_index=True,
        return_inverse=True,
    )

    sample_coordinates = coords[unique_map]
    coordinates = [torch.from_numpy(sample_coordinates).int()]
    sample_features = colors[unique_map]
    features = [torch.from_numpy(sample_features).float()]
    raw_coordinates = torch.from_numpy(points[unique_map]).float()

    coordinates, _ = ME.utils.sparse_collate(coords=coordinates, feats=features)
    features = torch.cat(features, dim=0)
    data = ME.SparseTensor(
        coordinates=coordinates,
        features=features,
        device=device,
    )
    
    if point2segment is not None:
        point2segment_full = point2segment.copy()
        _, _, ret_inv = np.unique(point2segment_full, return_index=True, return_inverse=True)
        point2segment_full = torch.from_numpy(ret_inv).unsqueeze(0)

        point2segment = point2segment[unique_map]
        _, _, ret_inv = np.unique(point2segment, return_index=True, return_inverse=True)
        point2segment = torch.from_numpy(ret_inv).unsqueeze(0).to(device)
    else:
        point2segment_full = None

    return data, point2segment, point2segment_full, raw_coordinates, inverse_map
