"""
For research only
"""

import os
import re
import glob
import torch
import PIL
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


def extract_features(args):

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
    transform_mask = _test_dataloader.dataset.transform_mask

    save_train_dir_path = os.path.join(
        "./artifacts", args.data_name, args.config_name, _train_dataloader.name
    )
    extracted_path = os.path.join(save_train_dir_path, "extracted.pt")

    if not os.path.exists(extracted_path):

        backbone = get_backbone(
            args.config.model.backbone_names[0]
        )  # FIXME: fix the hard-coding

        feature_embedder = FeatureEmbedder(
            device,
            input_shape,
            backbone,
            config.model.layers_to_extract,
            embedding_to_extract_from="avgpool",
        )

        fea_map_shape = feature_embedder.get_feature_map_shape()

        mapper = torch.nn.Linear(
            fea_map_shape[-1], 128, bias=False  # hard coded, always fixed
        )

        freeze(mapper)

        mapper = mapper.to(device)

        feas = []
        reduced_feas = []
        gaps = []
        masks = []
        downsized_masks = []
        labels = []
        class_sizes = []
        class_names = []
        image_paths = []

        for data in tqdm(_train_dataloader, desc="Extracting features..."):
            _images = data["image"]
            _image_paths = data["image_path"]

            _labels = data["is_anomaly"]

            _class_sizes = get_class_sizes_mvtec(_image_paths)
            _class_names = _get_class_names_mvtec(_image_paths)

            _masks = data["mask"]
            _masks, _labels = revise_masks_mvtec(
                _image_paths, _masks, _labels, transform_mask
            )

            _downsized_masks = _resize_mask(
                _masks, target_size=(fea_map_shape[0], fea_map_shape[1])
            )

            # # FIXME: patchify or not
            # _resized_masks = torch.round(_resized_masks)
            # _patchified_masks = _patchify(_resized_masks)

            # _patchified_ceil_masks = torch.ceil(_patchified_masks)
            # _patchified_round_masks = torch.round(_patchified_masks)

            # _up_resized_masks = _upscale(_resized_masks, _masks.shape[-1])
            # _up_patchified_ceil_masks = _upscale(_patchified_ceil_masks, _masks.shape[-1])
            # _up_patchified_round_masks = _upscale(_patchified_round_masks, _masks.shape[-1])

            # _patchified_ceil_masks = _patchified_ceil_masks[:, 0, :, :].to(torch.uint8)
            # _patchified_round_masks = _patchified_round_masks[:, 0, :, :].to(torch.uint8)

            # _up_resized_masks = _up_resized_masks[:, 0, :, :]
            # _up_patchified_ceil_masks = _up_patchified_ceil_masks[:, 0, :, :]
            # _up_patchified_round_masks = _up_patchified_round_masks[:, 0, :, :]

            _feas, _gaps = feature_embedder(_images, return_embeddings=True)

            with torch.no_grad():
                _reduced_feas = mapper(_feas.to(device)).cpu()

            _feas = unpatch_feas(_feas, fea_map_shape=fea_map_shape[:2])
            _reduced_feas = unpatch_feas(_reduced_feas, fea_map_shape=fea_map_shape[:2])

            feas.append(_feas)
            reduced_feas.append(_reduced_feas)
            gaps.append(_gaps)
            masks.append(_masks)
            downsized_masks.append(_downsized_masks)
            labels.append(_labels)
            class_sizes.append(_class_sizes)
            class_names += _class_names
            image_paths += _image_paths

            # for i in range(len(_masks)):
            #     if _masks[i].sum().item() > 0:
            #         _plot_and_save_tensor(_masks[i], "./_masks.png")
            #         _plot_and_save_tensor(_resized_masks[i], "./_resized_masks.png")
            #         _plot_and_save_tensor(_patchified_ceil_masks[i], "./_patchified_ceil_mask.png")
            #         _plot_and_save_tensor(_patchified_round_masks[i], "./_patchified_round_masks.png")
            #         _plot_and_save_tensor(_up_resized_masks[i], "./_up_resized_masks.png")
            #         _plot_and_save_tensor(_up_patchified_ceil_masks[i], "./_up_patchified_ceil_masks.png")
            #         _plot_and_save_tensor(_up_patchified_round_masks[i], "./_up_patchified_round_masks.png")

            #         _diff_resize = (_up_resized_masks[i] - _masks[i])
            #         _diff_patchify_ceil = (_up_patchified_ceil_masks[i] - _masks[i])
            #         _diff_patchify_round = (_up_patchified_round_masks[i] - _masks[i])

            #         _diff_resize = torch.norm(_diff_resize, p=1)
            #         _diff_patchify_ceil = torch.norm(_diff_patchify_ceil, p=1)
            #         _diff_patchify_round = torch.norm(_diff_patchify_round, p=1)

            #         # _diff_resize = _diff_resize[_diff_resize<0].sum()
            #         # _diff_patchify_ceil = _diff_patchify_ceil[_diff_patchify_ceil<0].sum()
            #         # _diff_patchify_round = _diff_patchify_round[_diff_patchify_round<0].sum()

            #         print(f"_diff_resize: {_diff_resize}")
            #         print(f"_diff_patchify_ceil: {_diff_patchify_ceil}")
            #         print(f"_diff_patchify_round: {_diff_patchify_round}")

        feas = torch.cat(feas)
        reduced_feas = torch.cat(reduced_feas)
        gaps = torch.cat(gaps)
        masks = torch.cat(masks)
        downsized_masks = torch.cat(downsized_masks)
        labels = torch.cat(labels)
        class_sizes = torch.cat(class_sizes)

        # # plotting test
        # anomaly_idx = torch.where(labels == 1)[0].tolist()[1]
        # mask = masks[anomaly_idx][0]
        # downsized_mask = downsized_masks[anomaly_idx][0]
        # image_path = image_paths[anomaly_idx]

        # _plot_and_save_tensor(mask, './_mask.png')
        # _plot_and_save_tensor(downsized_mask, './_downsized_mask.png')

        print("Saving features...")
        os.makedirs(os.path.dirname(extracted_path), exist_ok=True)
        torch.save(
            {
                "feas": feas,
                "reduced_feas": reduced_feas,
                "masks": masks,
                "downsized_masks": downsized_masks,
                "gaps": gaps,
                "labels": labels,
                "class_sizes": class_sizes,
                "class_names": class_names,
            },
            extracted_path,
        )


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

    pattern = os.path.join(os.path.dirname(image_path), "*.png")

    png_files = glob.glob(pattern)

    return len(png_files)


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
    extract_features(args)
