"""
For research only
"""

import os
import torch
import time
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
from src.sampler import LOFSampler, TailSampler, FewShotLOFSampler


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
    save_train_dir_path = os.path.join(
        "./artifacts", args.data_name, args.config_name, _train_dataloader.name
    )
    extracted_path = os.path.join(save_train_dir_path, "extracted.pt")

    assert os.path.exists(extracted_path)

    extracted = torch.load(extracted_path)

    feas = extracted["feas"]
    masks = extracted["masks"]

    gaps = extracted["gaps"]
    labels = extracted["labels"]
    class_names = extracted["class_names"]
    class_sizes = extracted["class_sizes"]

    # analyze_gap(
    #     gaps, masks, class_names, class_sizes, save_train_dir_path, save_plot=False
    # )
    analyze_patch(feas, masks, class_names, save_train_dir_path, save_plot=False)


def analyze_gap(gaps, masks, class_names, class_sizes, save_dir, save_plot=False):

    if gaps.ndim == 4:
        gaps = gaps[:, :, 0, 0]

    class_labels, class_label_names = _convert_class_names_to_labels(class_names)
    is_anomaly_gt = (masks.sum(dim=(1, 2, 3)) > 0).to(torch.long)

    _evaluate_tail_class_detection(
        gaps=gaps,
        class_sizes_gt=class_sizes,
        is_anomaly_gt=is_anomaly_gt,
        save_dir=save_dir,
    )

    # FIXME:
    save_plot = False
    if save_plot:

        save_plot_dir = os.path.join(save_dir, "plot_gap")

        self_sim = class_size.compute_self_sim(gaps)
        _anomaly_labels = is_anomaly_gt.to(torch.bool).tolist()
        _few_shot_labels = (class_sizes < 20).tolist()

        self_sim_abnormal_plot_dir = os.path.join(save_plot_dir, "self_sim_abnormal")
        self_sim_few_shot_plot_dir = os.path.join(save_plot_dir, "self_sim_few_shot")
        self_sim_else_plot_dir = os.path.join(save_plot_dir, "self_sim_else")
        os.makedirs(self_sim_abnormal_plot_dir, exist_ok=True)
        os.makedirs(self_sim_few_shot_plot_dir, exist_ok=True)
        os.makedirs(self_sim_else_plot_dir, exist_ok=True)

        def plot_gap_self_sim(index):
            print(f"plotting ngap for {index}")
            _scores = self_sim[index]
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
                th=np.cos(np.pi/4),
                filename=_filename,
            )

        Parallel(n_jobs=-1)(delayed(plot_gap_self_sim)(i) for i in range(len(self_sim)))


def analyze_patch(
    feas: torch.Tensor, masks, class_names, save_dir, save_plot=False
):

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
        is_anomaly_patch_gt, features, feature_map_shape, save_dir
    )

    # FIXME: bug fix is required
    save_plot = False
    if save_plot:
        save_dir_normal = os.path.join(save_dir, "self_sim", "normal")
        save_dir_abnormal = os.path.join(save_dir, "self_sim", "abnormal")

        os.makedirs(save_dir_normal, exist_ok=True)
        os.makedirs(save_dir_abnormal, exist_ok=True)

        th = np.cos(np.pi / 8)  # FIXME: tmp
        pixelwise_gt_scores = pixelwise_gt_scores.numpy()
        pixelwise_labels = pixelwise_labels.numpy()
        pixelwise_self_sim = class_size.compute_self_sim(pixelwise_feas).numpy()

        def plot_pixelwise_self_sim(p, i):
            print(f"Plotting (p,i) = ({p}, {i})")

            _self_sim = pixelwise_self_sim[p]
            _scores = _self_sim[i]
            _gt_scores = pixelwise_gt_scores[p].astype(np.float_)
            _labels = pixelwise_labels[p].astype(np.int_)

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
            for p in range(len(pixelwise_labels))
            for i in range(len(pixelwise_self_sim[p]))
        ]

        Parallel(n_jobs=-1)(delayed(plot_pixelwise_self_sim)(p, i) for p, i in pi_pairs)


def _evaluate_anomaly_patch_detection(
    is_anomaly_patch_gt, features, feature_map_shape, save_dir
) -> dict:

    # method_names = ["scs-indep-mean", "lof", "lof-scs-indep-mean"]
    method_names = ["lof-scs-indep-mean"]
    # method_names = ["lof"]
    results = []
    for method_name in method_names:
        _result = _get_result_anomaly_patch_detection(
            method_name, is_anomaly_patch_gt, features, feature_map_shape
        )
        results.append(_result)

    df = utils.save_dicts_to_csv(
        results, filename=os.path.join(save_dir, "result_patch.csv")
    )
    utils.print_df(df)


def _get_result_anomaly_patch_detection(
    method_name, is_anomaly_patch_gt, features, feature_map_shape
):

    if method_name == "scs-indep-mean":
        _, scs_idxes = class_size.sample_few_shot(
            features, feature_map_shape, th_type="indep"
        )
        is_anomaly_patch_pred = convert_indices_to_bool(len(features), scs_idxes)
    elif method_name == "lof":
        _, lof_idxes = LOFSampler().run(features, feature_map_shape)
        is_anomaly_patch_pred = 1 - convert_indices_to_bool(len(features), lof_idxes)
    elif method_name == "lof-scs-indep-mean":
        _, lofcsp_idxes = FewShotLOFSampler().run(
            features, feature_map_shape,
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


def _evaluate_tail_class_detection(
    gaps: torch.Tensor,
    class_sizes_gt: torch.Tensor,
    is_anomaly_gt: torch.Tensor,
    save_dir: str,
):

    # method_names = [
    #     "lof",
    #     "lof_scs-indep-mean",
    #     "scs-symmin-mean",
    #     "scs-symmin-mode",
    #     "scs-symmin-nearest",
    #     "scs-symavg-mean",
    #     "scs-symavg-mode",
    #     "scs-symavg-nearest",
    #     "scs-indep-mean",
    #     "scs-indep-mode",
    #     "scs-indep-nearest",
    # ]

    # method_names = [
    #     "lof_scs-indep-mean",
    # ]

    method_names = [
        "scs-indep-mean",
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
    
    method_parts = method_name.split("-")
    method_class = method_parts[0]

    if method_name == "lof":
        lof_sampler = LOFSampler()
        _, head_indices = lof_sampler.run(gaps)
        is_head_pred = convert_indices_to_bool(len(gaps), head_indices)
        is_tail_pred = 1 - is_head_pred
    elif method_name == "lof_scs-indep-mean":
        lof_sampler = LOFSampler()
        _, head_indices = lof_sampler.run(gaps, augment_class_sizes=True)
        is_head_pred = convert_indices_to_bool(len(gaps), head_indices)
        is_tail_pred = 1 - is_head_pred
    elif method_class == "scs":
        th_type = method_parts[1]
        vote_type = method_parts[2]
        tail_sampler = TailSampler(th_type=th_type, vote_type=vote_type)
        _, tail_indices = tail_sampler.run(gaps)
        is_tail_pred = convert_indices_to_bool(len(gaps), tail_indices)
    else:
        raise NotImplementedError()

    is_tail_gt = (class_sizes_gt <= 20).to(torch.long)

    is_missing_tail = (is_tail_pred - is_tail_gt) < 0
    is_included_anomaly = (is_tail_pred + is_anomaly_gt) > 1

    num_missing_tail = is_missing_tail.sum().item()
    num_included_anomaly = is_included_anomaly.sum().item()

    return {
        "method": method_name,
        "num_missing_tail": num_missing_tail,
        "num_included_anomaly": num_included_anomaly,
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

    return class_labels, class_label_names


def _get_result_csc(
    gaps,
    class_sizes_gt,
    anomaly_labels: torch.Tensor,
    vote_type: str,
    th_type: str,
):

    _, few_shot_indices = class_size.sample_few_shot(
        gaps, th_type=th_type, vote_type=vote_type
    )
    _result = _get_few_shot_result(
        len(gaps), few_shot_indices, class_sizes_gt, anomaly_labels
    )

    _result["method"] = "csc"
    _result["th_type"] = th_type
    _result["vote_type"] = vote_type

    return _result


def convert_indices_to_bool(n: int, indices: torch.Tensor) -> torch.Tensor:
    bool_array = torch.zeros((n), dtype=torch.long)
    bool_array[indices] = 1
    return bool_array


def _get_result_lof(
    gaps,
    class_sizes_gt,
    anomaly_labels: torch.Tensor,
    augment_class_sizes: bool = False,
):
    lof_sampler = LOFSampler()
    _, many_shot_indices = lof_sampler.run(
        gaps, augment_class_sizes=augment_class_sizes
    )

    def complementary_indices(indices: torch.Tensor, n: int):
        # Create a tensor of all indices
        all_indices = torch.arange(n)

        # Find the complementary indices
        mask = ~torch.isin(all_indices, indices)

        # Apply the mask to get complementary indices
        complement = all_indices[mask]

        return complement

    few_shot_indices = complementary_indices(many_shot_indices, len(gaps))

    _result = _get_few_shot_result(
        len(gaps), few_shot_indices, class_sizes_gt, anomaly_labels
    )

    _result["method"] = "lof"

    if augment_class_sizes:
        _result["method"] = "lof-csc"

    return _result


def _get_few_shot_result(
    n_samples,
    few_shot_indices: torch.Tensor,
    class_sizes_gt: torch.Tensor,
    anomaly_labels: torch.Tensor,
):
    is_few_shot_pred = torch.zeros((n_samples))
    is_few_shot_pred[few_shot_indices] = 1
    is_few_shot_target = (class_sizes_gt <= 20).to(torch.long)

    is_missing_few_show = (is_few_shot_pred - is_few_shot_target) < 0
    is_included_anomaly = (is_few_shot_pred + anomaly_labels) > 1

    num_missing_few_shot_samples = is_missing_few_show.sum().item()
    num_included_anomaly_samples = is_included_anomaly.sum().item()

    return {
        "num_included_anomaly_samples": num_included_anomaly_samples,
        "num_missing_few_shot_samples": num_missing_few_shot_samples,
    }


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
