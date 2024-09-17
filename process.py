# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2  # type: ignore

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import argparse
import os
from typing import Any, Dict, List
import numpy as np
import matplotlib.pyplot as plt
import torch
from copy import deepcopy
import copy
import open3d as o3d
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--input",
    type=str,
    # required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    # required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)

amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['predicted_iou']), reverse=False)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.45]])
        img[m] = color_mask
    ax.imshow(img)


def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return

def get_sam(image, mask_generator):
    masks, masks_s, masks_m, masks_l = mask_generator.generate(image)
    group_ids = return_group_ids(image, masks)
    group_ids_s = return_group_ids(image, masks_s)
    group_ids_m = return_group_ids(image, masks_m)
    group_ids_l = return_group_ids(image, masks_l)
    return group_ids, group_ids_s, group_ids_m, group_ids_l

def return_group_ids(image, masks):
    group_ids = np.full((image.shape[0], image.shape[1]), -1, dtype=int)
    num_masks = len(masks)
    group_counter = 0
    for i in reversed(range(num_masks)):
        if isinstance(masks[i], dict):
            group_ids[masks[i]["segmentation"]] = group_counter
        else:
            group_ids[masks[i]] = group_counter
        group_counter += 1
    return group_ids

def num_to_natural(group_ids):
    '''
    Change the group number to natural number arrangement
    '''
    if np.all(group_ids == -1):
        return group_ids
    array = copy.deepcopy(group_ids)
    unique_values = np.unique(array[array != -1])
    mapping = np.full(np.max(unique_values) + 2, -1)
    mapping[unique_values + 1] = np.arange(len(unique_values))
    array = mapping[array + 1]
    return array

def visualize_partition(coord, group_id, save_path):
    group_id = group_id.reshape(-1)
    num_groups = group_id.max() + 1
    group_colors = np.random.rand(num_groups, 3)
    group_colors = np.vstack((group_colors, np.array([0,0,0])))
    color = group_colors[group_id]
    save_point_cloud(coord, color, save_path)

def save_point_cloud(coord, color=None, file_path="pc.ply", logger=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    coord = to_numpy(coord)
    if color is not None:
        color = to_numpy(color)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(np.ones_like(coord) if color is None else color)
    o3d.io.write_point_cloud(file_path, pcd)
    if logger is not None:
        logger.info(f"Save Point Cloud to: {file_path}")

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().cpu().numpy()
    assert isinstance(x, np.ndarray)
    return x

def extract_masks(mask_array):
    unique_values = np.unique(mask_array)
    unique_values = unique_values[unique_values != -1]

    masks = []
    for val in unique_values:
        mask = np.where(mask_array == val, True, False)
        masks.append(mask)

    return masks

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def find_contained_masks(masks, masks_l, iou=0.8):
    contained_masks_indices = []

    for i, mask in enumerate(masks):
        is_contained = False
        for mask_l in masks_l:
            intersection = np.sum(np.logical_and(mask, mask_l))
            if intersection/np.sum(mask) > iou:
                is_contained = True
                break

        if is_contained:
            contained_masks_indices.append(i)

    return contained_masks_indices

def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs

def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam = sam.to(device=args.device)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    print(amg_kwargs)
    mask_generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)

    data_dir = 'PATH_TO_SCANNET'
    scenes = sorted([folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))])

    for scene in tqdm(scenes):
        scene_folder = os.path.join(data_dir, scene)
        sam_folder = os.path.join(scene_folder, 'sam')
        os.makedirs(sam_folder, exist_ok=True)

        rgb_folder = os.path.join(scene_folder, 'color')
        rgb_images = [file for file in os.listdir(rgb_folder) if file.endswith('.png')]
        
        for image_file in rgb_images:
            image_id = image_file.split('.')[0]

            color_path = os.path.join(rgb_folder, image_file)
            color_image = cv2.imread(color_path)
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            color_image = cv2.resize(color_image, (640, 480))

            sam_groups, sam_s_groups, sam_m_groups, sam_l_groups = get_sam(color_image, mask_generator)
            sam_masks = extract_masks(sam_groups)
            sam_l_masks = extract_masks(sam_l_groups)

            contained_indices = find_contained_masks(sam_masks, sam_l_masks, iou=0.9)
            sam_whole_masks = [sam_masks[i] for i in range(len(sam_masks)) if i not in contained_indices]
            sam_whole_masks.extend(sam_l_masks)
            sam_whole_groups = return_group_ids(color_image, sam_whole_masks)

            sam_image_path = os.path.join(sam_folder, f'{image_id}.png')
            img = Image.fromarray(num_to_natural(sam_whole_groups).astype(np.int16), mode='I;16')
            img.save(sam_image_path)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)