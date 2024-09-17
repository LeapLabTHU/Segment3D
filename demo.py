import torch
import pyviz3d.visualizer as viz
import numpy as np
from pathlib import Path
from hydra.experimental import initialize, compose
import hydra
from omegaconf import OmegaConf, DictConfig
import os
from cuml.cluster import DBSCAN
import json
from demo_utils import get_model, load_mesh, prepare_data
from torch_scatter import scatter_mean

@hydra.main(
    config_path="conf", config_name="config_base_instance_segmentation.yaml"
)
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    model = get_model(cfg)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # load input data
    pointcloud_file = f"demo_test/{cfg.general.test_scene}/mesh.ply"
    mesh = load_mesh(pointcloud_file)

    point2segment = None
    if cfg.general.train_on_segments:
        segment_filepath = f"demo_test/{cfg.general.test_scene}/mesh.0.010000.segs.json"
        with open(segment_filepath) as f:
            segments = json.load(f)
            point2segment = np.array(segments["segIndices"])

    # prepare data
    data, point2segment, point2segment_full, raw_coordinates, inverse_map = prepare_data(cfg, mesh, point2segment, device)

    # run model
    with torch.no_grad():
        outputs = model(data, point2segment=point2segment, raw_coordinates=raw_coordinates)

    # parse predictions
    scores, masks_binary = parse_predictions(cfg, outputs, point2segment, point2segment_full, raw_coordinates, inverse_map)

    # filter masks with a confidence threshold
    save_visualization(mesh, scores, masks_binary, cfg.general.test_scene, confidence_threshold=0.2)


def get_mask_and_scores(cfg, mask_cls, mask_pred):

    result_pred_mask = (mask_pred > 0).float()

    mask_pred = mask_pred[:,result_pred_mask.sum(0)>0]
    mask_cls = mask_cls[result_pred_mask.sum(0)>0]
    result_pred_mask = result_pred_mask[:,result_pred_mask.sum(0)>0]
    heatmap = mask_pred.float().sigmoid()

    mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (
        result_pred_mask.sum(0) + 1e-6
    )
    score = mask_cls * mask_scores_per_image

    topk_count = min(cfg.general.topk_per_image, len(score)) if cfg.general.topk_per_image != -1 else len(score)
    score, topk_indices = score.topk(topk_count, sorted=True)

    result_pred_mask = result_pred_mask[:, topk_indices]
    return score, result_pred_mask


def get_full_res_mask(mask, inverse_map, point2segment_full):
    mask = mask[inverse_map]  # full res
    if point2segment_full is not None:
        mask = scatter_mean(mask, point2segment_full.squeeze(0), dim=0)  # full res segments
        mask = (mask > 0.5).float()
        mask = mask[point2segment_full.squeeze(0)]  # full res points
    return mask


def parse_predictions(cfg, outputs, point2segment, point2segment_full, raw_coordinates, inverse_map):
    logits = outputs["pred_logits"][0][:,0].detach().cpu()
    masks = outputs["pred_masks"][0].detach().cpu()

    if cfg.model.train_on_segments:
        masks = outputs["pred_masks"][0].detach().cpu()[point2segment.cpu()].squeeze(0)
    else:
        masks = outputs["pred_masks"][0].detach().cpu()

    if cfg.general.use_dbscan:
        new_logits = []
        new_masks = []
        for curr_query in range(masks.shape[1]):
            curr_masks = masks[:, curr_query] > 0
            if raw_coordinates[curr_masks].shape[0] > 0:
                clusters = (
                    DBSCAN(
                        eps=cfg.general.dbscan_eps,
                        min_samples=cfg.general.dbscan_min_points,
                        verbose=2
                    )
                    .fit(raw_coordinates[curr_masks].cuda())
                    .labels_
                )
                clusters = clusters.get()
                new_mask = np.zeros(curr_masks.shape, dtype=int)
                new_mask[curr_masks] = clusters + 1

                for cluster_id in np.unique(clusters):
                    original_pred_masks = masks[:, curr_query].numpy()
                    if cluster_id != -1:
                        if (new_mask == cluster_id + 1).sum() > cfg.data.remove_small_group:
                            new_logits.append(logits[curr_query])
                            new_masks.append(
                                torch.from_numpy(original_pred_masks
                                * (new_mask == cluster_id + 1))
                            )
        logits = new_logits
        masks = new_masks

    scores, masks = get_mask_and_scores(
        cfg,
        torch.stack(logits).cpu(),
        torch.stack(masks).T,
    )

    masks_binary = get_full_res_mask(masks, inverse_map, point2segment_full)
    masks_binary = masks_binary.permute(1,0).bool()
    return scores, masks_binary


def save_visualization(mesh, scores, masks_binary, scene_name, confidence_threshold):
    v = viz.Visualizer()
    point_positions = np.asarray(mesh.vertices)
    point_colors = np.asarray(mesh.vertex_colors)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    point_normals = np.asarray(mesh.vertex_normals)

    v.add_points('rgb', point_positions, point_colors * 255, point_normals, point_size=20)
    pred_coords = []
    pred_inst_color = []
    pred_normals = []
    for i in reversed(range(len(masks_binary))):
        mask_i = masks_binary[i]
        score_i = scores[i]
        if score_i > confidence_threshold:
            num_i = point_positions[mask_i].shape[0]
            color_i = np.tile(np.random.rand(3) * 255, [num_i, 1])
            v.add_points(f'{i}_{score_i:.2f}',
                        point_positions[mask_i],
                        color_i,
                        point_normals[mask_i], point_size=20, visible=False)
            pred_coords.append(point_positions[mask_i])
            pred_inst_color.append(color_i)
            pred_normals.append(point_normals[mask_i])
    pred_coords = np.concatenate(pred_coords)
    pred_inst_color = np.concatenate(pred_inst_color)
    pred_normals = np.concatenate(pred_normals)
    v.add_points(
            "Instances",
            pred_coords,
            pred_inst_color,
            pred_normals, point_size=20, alpha=0.8, visible=False)
    output_path = f'demo_test/{scene_name}/visualization'
    print(output_path)
    v.save(output_path)

if __name__ == "__main__":
    main()
