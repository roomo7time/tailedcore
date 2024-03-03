"""
For research only
"""

import os
import re
import glob
import torch
import PIL
import argparse
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms


import src.evaluator.result as result
from src import utils
from src.dataloader import get_dataloaders
from src.get_args import parse_args  # FIXME: make independent args
from src.engine import Engine
from src.backbone import get_backbone
from src.feature_embedder import FeatureEmbedder
from src.feature_extractor import FeatureExtractor

from src.patch_maker import PatchMaker


def extract_artifacts(args):
    args.batch_size = 64
    assert args.data_format == "mvtec-multiclass"

    utils.set_seed(args.config.seed)

    config = args.config

    device = utils.set_torch_device(args.gpu)

    input_shape = (3, config.data.inputsize, config.data.inputsize)

    dataloaders = get_dataloaders(
        config.data,
        data_format=args.data_format,
        data_path=args.data_path,
        batch_size=args.batch_size,
    )

    _train_dataloader = dataloaders[0]["train"]
    _test_dataloader = dataloaders[0]["test"]

    save_extracted_dir = os.path.join("./artifacts", args.data_name, args.config_name)

    extracted_path_train = os.path.join(
        save_extracted_dir, f"extracted_train_{_train_dataloader.name}.pt"
    )

    if os.path.exists(extracted_path_train):
        return

    backbone = get_backbone(args.config.model.backbone_names[0])

    feature_extractor = FeatureEmbedder(
        device,
        input_shape,
        backbone,
        config.model.layers_to_extract,
        embedding_to_extract_from=config.model.embedding_to_extract_from,
    )

    feature_extractor.eval()
    feature_map_shape = feature_extractor.get_feature_map_shape()

    
    transform_mask = _test_dataloader.dataset.transform_mask
    
    artifacts = _extract_artifacts(_train_dataloader, feature_extractor, transform_mask)

    artifacts["feature_map_shape"] = feature_map_shape

    print("Saving artifacts...")
    os.makedirs(os.path.dirname(extracted_path_train), exist_ok=True)
    torch.save(artifacts, extracted_path_train)
    print("Artifacts have been saved...")


def _extract_artifacts(
    dataloader,
    feature_extractor,
    transform_mask,
):
    print("Extracting artifacts...")
    feas = []
    gaps = []
    masks = []
    labels = []
    class_sizes = []
    class_names = []
    image_names = []

    for data in tqdm(dataloader, desc="Extracting features..."):
        _images = data["image"]
        _image_paths = data["image_path"]
        _image_names = data["image_name"]

        _labels = data["is_anomaly"]

        _class_sizes = get_class_sizes_mvtec(_image_paths)
        _class_names = _get_class_names_mvtec(_image_paths)

        _masks = data["mask"]

        _masks, _labels = revise_masks_mvtec(
            _image_paths, _masks, _labels, transform_mask
        )

        _feas, _gaps = feature_extractor(_images, return_embeddings=True)

        feas.append(_feas)
        gaps.append(_gaps)
        masks.append(_masks)
        labels.append(_labels)
        class_sizes.append(_class_sizes)
        class_names += _class_names
        image_names += _image_names

    feas = torch.cat(feas)
    gaps = torch.cat(gaps)
    masks = torch.cat(masks)
    labels = torch.cat(labels)
    class_sizes = torch.cat(class_sizes)

    return {
        "feas": feas,
        "masks": masks,
        "gaps": gaps,
        "labels": labels,
        "class_sizes": class_sizes,
        "class_names": class_names,
    }


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def unpatch_feas(feas, fea_map_shape):
    fea_dim = feas.shape[-1]
    h, w = fea_map_shape[0], fea_map_shape[1]
    feas = feas.reshape(-1, h * w, fea_dim)
    b = feas.shape[0]
    feas = feas.permute(0, 2, 1)
    feas = feas.reshape(b, fea_dim, h, w)
    return feas


def _is_anomaly_path_mvtec(file_path):
    file_name_without_ext, _ = os.path.splitext(file_path)
    file_name_only = os.path.basename(file_name_without_ext)

    if re.fullmatch(r"\d+", file_name_only):
        return False
    else:
        return True


def _get_defect_type_name_mvtec(file_path):
    file_name_without_ext, _ = os.path.splitext(file_path)
    file_name_only = os.path.basename(file_name_without_ext)

    defect_type = file_name_only[:-4]
    return defect_type


from src.utils import modify_subfolders_in_path


def revise_masks_mvtec(image_paths, masks, labels, transform_mask):

    masks = masks.clone().detach()
    labels = labels.clone().detach()

    for i, image_path in enumerate(image_paths):
        if _is_anomaly_path_mvtec(image_path):

            # FIXME: refactor
            image_file_name_without_ext, ext = os.path.splitext(image_path)
            image_file_name_without_ext = image_file_name_without_ext[-3:]
            mask_file_name = f"{image_file_name_without_ext}_mask{ext}"
            mask_path = os.path.join(os.path.dirname(image_path), mask_file_name)

            mask_path = modify_subfolders_in_path(
                mask_path,
                {
                    -3: "ground_truth",
                    -2: _get_defect_type_name_mvtec(image_path),
                    -1: os.path.basename(mask_path),
                },
            )

            # TODO: debug
            mask_dir = os.path.dirname(mask_path)
            mask_path_pattern = f"{mask_dir}/*{image_file_name_without_ext}*"
            mask_path = glob.glob(mask_path_pattern)[0]

            mask = PIL.Image.open(mask_path)
            mask = transform_mask(mask)

            masks[i] = mask
            labels[i] = 1

    masks = masks.to(torch.uint8)
    masks = masks.to(torch.float32)

    return masks, labels


def get_class_sizes_mvtec(image_paths):
    class_sizes = torch.tensor(
        [_get_class_size_mvtec(image_path) for image_path in image_paths]
    )
    return class_sizes


def _get_class_size_mvtec(image_path):

    # pattern = os.path.join(os.path.dirname(image_path), "*.png")
    pattern = os.path.join(os.path.dirname(image_path), "*")

    files = glob.glob(pattern)

    return len(files)


def _resize_mask(masks, target_size, binarize=False):
    _resized_masks = F.interpolate(
        masks, mode="bilinear", size=(target_size[0], target_size[1])
    )
    if binarize:
        _resized_masks = torch.round(_resized_masks)
    return _resized_masks


def _get_class_names_mvtec(image_paths):
    return [_get_folder_name(image_path, -3) for image_path in image_paths]


def _get_folder_name(image_path, depth):
    # Split the path into parts
    path_parts = image_path.split(os.sep)

    # Get the folder name at the specified depth
    return path_parts[depth - 1]


def _plot_and_save_tensor(tensor, filename="output.png"):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")

    if tensor.ndim != 2:
        raise ValueError("Tensor must be 2-dimensional.")

    # Plotting the tensor
    plt.imshow(tensor, cmap="gray")
    plt.colorbar()

    # Saving the plot
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":

    args = parse_args()
    extract_artifacts(args)
