"""
For research only
"""

import os
import torch
import time
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from collections import Counter
from tqdm import tqdm
from joblib import Parallel, delayed

import src.evaluator.result as result
import src.class_size as class_size
import src.helpers.cv2plot as cv2plot

from src import utils
from src.dataloader import get_dataloaders
from src.get_args import parse_args  # FIXME: make independent args
from src.engine import Engine
from src.backbone import get_backbone
from src.feature_embedder import FeatureEmbedder

from src.patch_maker import PatchMaker
from src.sampler import LOFSampler, TailSampler, TailedLOFSampler


def analyze_extracted(args):
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
    save_extracted_dir = os.path.join(
        "./artifacts", args.data_name, args.config_name, _train_dataloader.name
    )
    extracted_path = os.path.join(save_extracted_dir, "extracted.pt")

    assert os.path.exists(extracted_path)

    extracted = torch.load(extracted_path)

    feas = extracted["feas"]
    masks = extracted["masks"]

    gaps = extracted["gaps"]
    labels = extracted["labels"]
    class_names = extracted["class_names"]
    class_sizes = extracted["class_sizes"]

    num_samples_per_class = dict(Counter(class_names))
    save_data_info_path = os.path.join(
        "./artifacts", args.data_name, "num_samples_per_class.csv"
    )
    utils.save_dicts_to_csv([num_samples_per_class], save_data_info_path)

    save_log_dir = os.path.join("./logs", f"{args.data_name}_{args.config_name}")

    analyze_gap(
        gaps, masks, class_names, class_sizes, save_log_dir, save_plot=False
    )
    analyze_patch(feas, masks, gaps, save_log_dir, save_plot=False)


def analyze_patch(feas: torch.Tensor, masks, gaps, save_dir, save_plot=False):

    downsized_masks = _downsize_masks(masks, mode="bilinear")

    if downsized_masks.ndim == 4:
        downsized_masks = downsized_masks[:, 0, :, :]

    n, fea_dim, h, w = feas.shape

    label_names = ["normal", "abnormal"]
    anomaly_patch_scores_gt = downsized_masks.reshape((-1))
    is_anomaly_patch_gt = torch.round(anomaly_patch_scores_gt).to(torch.long)
    feature_map_shape = [28, 28]
    features = (
        feas.reshape((feas.shape[0], feas.shape[1], -1))
        .permute(0, 2, 1)
        .reshape((-1, fea_dim))
    )

    _evaluate_anomaly_patch_detection(
        is_anomaly_patch_gt, features, feature_map_shape, gaps, save_dir
    )

    if save_plot:
        save_dir_normal = os.path.join(save_dir, "plot", "self_sim", "normal")
        save_dir_abnormal = os.path.join(save_dir, "plot", "self_sim", "abnormal")

        os.makedirs(save_dir_normal, exist_ok=True)
        os.makedirs(save_dir_abnormal, exist_ok=True)

        th = np.cos(np.pi / 8)  # FIXME: tmp
        pixelwise_feas = feas.reshape(n, fea_dim, -1).permute(2, 0, 1)
        pixelwise_gt_scores = anomaly_patch_scores_gt.reshape(n, h * w).numpy()
        pixelwise_labels = is_anomaly_patch_gt.reshape(n, h * w).numpy()
        pixelwise_self_sim = class_size.compute_self_sim(pixelwise_feas).numpy()

        def plot_pixelwise_self_sim(p, i):
            print(f"Plotting (p,i) = ({p}, {i})")

            _self_sim = pixelwise_self_sim[p]
            _scores = _self_sim[i]
            _gt_scores = pixelwise_gt_scores[:, p].astype(np.float_)
            _labels = pixelwise_labels[:, p].astype(np.int_)

            if _labels[i] > 0:
                _dir = save_dir_abnormal
                _filename = os.path.join(_dir, f"p{p:03d}_i{i:04d}.jpg")
            else:
                _dir = os.path.join(save_dir_normal, f"p{p:03d}")
                os.makedirs(_dir, exist_ok=True)
                _filename = os.path.join(_dir, f"i{i:04d}.jpg")

            cv2plot.plot_scatter(
                _scores,
                _labels,
                label_names,
                extra_labels=_labels,
                extra_scores=_gt_scores,
                th=th,
                filename=_filename,
            )

        pi_pairs = [
            (p, i)
            for p in range(len(pixelwise_self_sim))
            for i in range(len(pixelwise_self_sim[p]))
        ]

        Parallel(n_jobs=-1)(delayed(plot_pixelwise_self_sim)(p, i) for p, i in pi_pairs)
        # for p, i in pi_pairs:
        #     plot_pixelwise_self_sim(p, i)


def _evaluate_anomaly_patch_detection(
    is_anomaly_patch_gt, features, feature_map_shape, gaps, save_dir
) -> dict:
    
    method_names = ["lof", "lof-scs_symmin"]
    results = []
    for method_name in method_names:
        _result = _get_result_anomaly_patch_detection(
            method_name,
            is_anomaly_patch_gt,
            features,
            feature_map_shape,
            gaps,
        )
        results.append(_result)

    df = utils.save_dicts_to_csv(
        results, filename=os.path.join(save_dir, "result_patch.csv")
    )
    utils.print_df(df)


def _get_result_anomaly_patch_detection(
    method_name, is_anomaly_patch_gt, features, feature_map_shape, gaps
):

    if method_name == "lof":
        _, lof_idxes = LOFSampler().run(features, feature_map_shape)
        is_anomaly_patch_pred = 1 - convert_indices_to_bool(len(features), lof_idxes)
    elif method_name == "lof-scs_symmin":
        _, lofcsp_idxes = TailedLOFSampler(tail_th_type="symmin").run(
            features,
            feature_map_shape,
        )
        is_anomaly_patch_pred = 1 - convert_indices_to_bool(len(features), lofcsp_idxes)
    else:
        raise NotImplementedError()

    is_missing_anomaly_patch = (is_anomaly_patch_pred - is_anomaly_patch_gt) < 0
    is_missing_normal_patch = (is_anomaly_patch_pred - is_anomaly_patch_gt) > 0
    num_missing_anomaly_patch = is_missing_anomaly_patch.sum().item()
    num_missing_normal_patch = is_missing_normal_patch.sum().item()

    _result = {
        "method": method_name,
        "num_missing_anomaly_patch": num_missing_anomaly_patch,
        "num_missing_normal_patch": num_missing_normal_patch,
    }

    return _result


def analyze_gap(gaps, masks, class_names, class_sizes, save_dir, save_plot=False):

    if gaps.ndim == 4:
        gaps = gaps[:, :, 0, 0]

    class_labels, class_label_names = _convert_class_names_to_labels(class_names)
    class_labels = class_labels.numpy()

    is_anomaly_gt = (masks.sum(dim=(1, 2, 3)) > 0).to(torch.long)

    _evaluate_tail_class_detection(
        gaps=gaps,
        class_sizes_gt=class_sizes,
        is_anomaly_gt=is_anomaly_gt,
        save_dir=save_dir,
    )

    # FIXME:
    if save_plot:

        _anomaly_labels = is_anomaly_gt.to(torch.bool).tolist()
        _few_shot_labels = (class_sizes < 20).tolist()
        save_plot_dir = os.path.join(save_dir, "plot_gap")

        self_sim = class_size.compute_self_sim(gaps)

        self_sim_abnormal_plot_dir = os.path.join(save_plot_dir, "self_sim_abnormal")
        self_sim_few_shot_plot_dir = os.path.join(save_plot_dir, "self_sim_few_shot")
        self_sim_else_plot_dir = os.path.join(save_plot_dir, "self_sim_else")
        os.makedirs(self_sim_abnormal_plot_dir, exist_ok=True)
        os.makedirs(self_sim_few_shot_plot_dir, exist_ok=True)
        os.makedirs(self_sim_else_plot_dir, exist_ok=True)

        def plot_gap_self_sim(index):
            print(f"plotting ngap for {index}")
            _scores = self_sim[index].numpy()
            _is_anomaly = _anomaly_labels[index]
            _is_few_shot = _few_shot_labels[index]
            if _is_anomaly:
                _filename = os.path.join(self_sim_abnormal_plot_dir, f"{index:04d}.jpg")
            elif _is_few_shot:
                _filename = os.path.join(self_sim_few_shot_plot_dir, f"{index:04d}.jpg")
            else:
                _filename = os.path.join(self_sim_else_plot_dir, f"{index:04d}.jpg")
            plot_scatter(
                _scores,
                class_labels,
                class_label_names,
                is_anomaly_gt,
                th=np.cos(np.pi / 4),
                filename=_filename,
            )

        # Parallel(n_jobs=-1)(delayed(plot_gap_self_sim)(i) for i in range(len(self_sim)))

        for i in range(len(self_sim)):
            plot_gap_self_sim(i)


def _evaluate_tail_class_detection(
    gaps: torch.Tensor,
    class_sizes_gt: torch.Tensor,
    is_anomaly_gt: torch.Tensor,
    save_dir: str,
):

    method_names = [
        "lof",
        "scs_symmin",
        "scs_indep",
    ]

    results = []
    for method_name in method_names:
        _result = _get_result_tail_class_detection(
            method_name=method_name,
            class_sizes_gt=class_sizes_gt,
            is_anomaly_gt=is_anomaly_gt,
            gaps=gaps,
        )
        results.append(_result)

    df = utils.save_dicts_to_csv(
        results, filename=os.path.join(save_dir, "result_gap.csv")
    )

    utils.print_df(df)


def _get_result_tail_class_detection(
    method_name,
    class_sizes_gt,
    is_anomaly_gt,
    gaps,
):

    if method_name == "lof":
        lof_sampler = LOFSampler()
        _, head_indices, tail_scores = lof_sampler.run(gaps, return_outlier_scores=True)
        is_head_pred = convert_indices_to_bool(len(gaps), head_indices)
        is_tail_pred = 1 - is_head_pred
        class_sizes_pred = torch.zeros((len(gaps)))
    elif "scs" in method_name:
        method_parts = method_name.split("_")
        th_type = method_parts[1]
        tail_sampler = TailSampler(th_type=th_type, vote_type="mean")
        _, tail_indices, class_sizes_pred = tail_sampler.run(
            gaps, return_class_sizes=True
        )
        is_tail_pred = convert_indices_to_bool(len(gaps), tail_indices)
        tail_scores = 1 -class_sizes_pred/class_sizes_pred.max()
    else:
        raise NotImplementedError()

    is_tail_gt = (class_sizes_gt <= 20).to(torch.long)

    is_missing_tail = (is_tail_pred - is_tail_gt) < 0
    is_included_anomaly = (is_tail_pred + is_anomaly_gt) > 1

    num_missing_tail = is_missing_tail.sum().item()
    num_included_anomaly = is_included_anomaly.sum().item()
    class_size_pred_error = (
        (
            abs(class_sizes_pred - class_sizes_gt)
            * (1 / class_sizes_gt)
            * (1 / class_sizes_gt)
        )
        .sum()
        .item()
    )
    tail_pred_acc = 1 - abs(is_tail_gt - is_tail_pred).to(torch.float).mean().item()

    return {
        "method": method_name,
        "num_missing_tail": num_missing_tail,
        "num_included_anomaly": num_included_anomaly,
        "tail_pred_acc": tail_pred_acc,
        "class_size_pred_error": class_size_pred_error,
        "auroc": metrics.roc_auc_score(is_tail_gt, tail_scores)
    }


_patchmaker = PatchMaker(patchsize=3, stride=1)


def _patchify(masks):
    _shape = masks.shape
    masks = _patchmaker.patchify(masks)
    c = _shape[1]
    h = _shape[-2]
    w = _shape[-1]

    masks = masks.reshape(-1, *masks.shape[-3:])
    # masks = torch.mean(masks, dim=(-1, -2)).squeeze()
    masks = masks.reshape(len(masks), c, -1)
    masks = F.adaptive_avg_pool1d(masks, 1)[:, :, 0]

    # incorrect?
    # masks = masks.reshape(-1, c, h, w)

    # # correct?
    masks = masks.reshape(-1, h * w, c)
    masks = masks.permute(0, 2, 1)
    masks = masks.reshape(len(masks), c, h, w)

    return masks


def _downsize_masks(masks, mode="patchify_last_depth"):
    if mode == "patchify_last_depth":
        masks = _downsize_masks_patchify_last_depth(masks)
    elif mode == "patchify_all_depths":
        masks = _downsize_mask_patchify_all_depths(masks)
    elif mode == "bilinear":
        _, _, h, w = masks.shape
        assert h == w
        depth = 2
        masks = F.interpolate(
            masks, size=h // (2 ** (depth + 1)), mode="bilinear", align_corners=False
        )

    return masks


def _downsize_masks_patchify_last_depth(masks, depth=2):
    _, _, h, w = masks.shape
    assert h == w

    masks = F.interpolate(
        masks, size=h // (2 ** (depth + 1)), mode="bilinear", align_corners=False
    )
    masks = _patchify(masks)

    return masks


def _downsize_mask_patchify_all_depths(masks, depth=2):
    _, _, h, w = masks.shape
    assert h == w

    for d in range(depth + 1):
        masks = F.interpolate(
            masks, size=h // (2 ** (d + 1)), mode="bilinear", align_corners=False
        )
        masks = _patchify(masks)

    return masks


def _convert_class_names_to_labels(class_names):
    label_map = {}
    class_labels = []
    current_label = 0

    for class_name in class_names:
        if class_name not in label_map:
            label_map[class_name] = current_label
            current_label += 1
        class_labels.append(label_map[class_name])

    class_label_names = list(label_map.keys())

    return torch.LongTensor(class_labels), class_label_names

def convert_indices_to_bool(n: int, indices: torch.Tensor) -> torch.Tensor:
    bool_array = torch.zeros((n), dtype=torch.long)
    bool_array[indices] = 1
    return bool_array






def plot_scatter(scores, labels, label_names, extra_labels, th, filename):

    cv2plot.plot_scatter(
        scores, labels, label_names, extra_labels, th=th, filename=filename
    )


def _plot_tensor(tensor, filename="output.png"):
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
    analyze_extracted(args)
