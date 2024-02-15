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
from src.sampler import LOFSampler


def analyze_extracted(args):
    utils.set_seed(args.config.seed)

    config = args.config

    device = utils.set_torch_device(args.gpu)

    input_shape = (3, config.data.inputsize, config.data.inputsize)

    dataloaders = get_dataloaders(
        config,
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

    gaps_dir = os.path.join(save_train_dir_path, "gaps")
    feas_dir = os.path.join(save_train_dir_path, "feas")

    # analyze_gaps(gaps, masks, class_names, class_sizes, gaps_dir, plot_self_sim=False)
    analyze_feas(feas, masks, class_names, feas_dir, plot_self_sim=False)


def analyze_feas(feas: torch.Tensor, masks, class_names, save_dir, plot_self_sim=False):

    save_dir_normal = os.path.join(save_dir, "self_sim", "normal")
    save_dir_abnormal = os.path.join(save_dir, "self_sim", "abnormal")

    os.makedirs(save_dir_normal, exist_ok=True)
    os.makedirs(save_dir_abnormal, exist_ok=True)

    downsized_masks = _downsize_masks(masks, mode="bilinear")

    if downsized_masks.ndim == 4:
        downsized_masks = downsized_masks[:, 0, :, :]

    n, fea_dim, h, w = feas.shape

    label_names = ["normal", "abnormal"]
    anomaly_gt_scores = downsized_masks.reshape((-1))
    is_anomaly_gt = torch.round(anomaly_gt_scores).to(torch.long)
    feature_map_shape = feas.shape[2:]
    features = (
        feas.reshape((feas.shape[0], feas.shape[1], -1)).permute(0, 2, 1).reshape((-1, fea_dim))
    )

    _, csc_idxes = class_size.sample_few_shot(
        features, feature_map_shape, th_type="indep"
    )
    _, lofcsp_idxes = LOFSampler().run(features, feature_map_shape, augment_class_sizes=True)
    _, lof_idxes = LOFSampler().run(features, feature_map_shape)
    

    # is_anomaly_csc = convert_indices_to_bool(len(features), csc_idxes)
    is_anomaly_lof = 1 - convert_indices_to_bool(len(features), lof_idxes)
    is_anomaly_lofcsp = 1 - convert_indices_to_bool(len(features), lofcsp_idxes)

    is_missing_anomaly_lof = (is_anomaly_lof - is_anomaly_gt) < 0
    is_missing_normal_lof = (is_anomaly_lof - is_anomaly_gt) > 0
    num_missing_anomaly_lof = is_missing_anomaly_lof.sum().item()
    num_missing_normal_lof = is_missing_normal_lof.sum().item()
    print(f"LOF - num. of missing anomaly pixels: {num_missing_anomaly_lof}")
    print(f"LOF - num. of missing normal pixels: {num_missing_normal_lof}")

    is_missing_anomaly_lofcsp = (is_anomaly_lofcsp - is_anomaly_gt) < 0
    is_missing_normal_lofcsp = (is_anomaly_lofcsp - is_anomaly_gt) > 0
    num_missing_anomaly_lofcsp = is_missing_anomaly_lofcsp.sum().item()
    num_missing_normal_lofcsp = is_missing_normal_lofcsp.sum().item()
    print(f"lofcsp - num. of missing anomaly pixels: {num_missing_anomaly_lofcsp}")
    print(f"lofcsp - num. of missing normal pixels: {num_missing_normal_lofcsp}")

    

    if plot_self_sim:
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


def analyze_gaps(gaps, masks, class_names, class_sizes, save_dir, plot_self_sim=False):

    if gaps.ndim == 4:
        gaps = gaps[:, :, 0, 0]

    class_labels, class_label_names = _convert_class_names_to_labels(class_names)
    anomaly_labels = (masks.sum(dim=(1, 2, 3)) > 0).to(torch.long)

    th_types = ["sym-min", "sym-avg", "random-approx", "indep"]
    vote_types = ["mean", "mode", "mean-nearest"]
    results = []

    results.append(
        _get_result_lof(gaps, class_sizes, anomaly_labels, augment_class_sizes=True)
    )
    results.append(_get_result_lof(gaps, class_sizes, anomaly_labels))

    for th_type in th_types:
        for vote_type in vote_types:
            print(f"th_type: {th_type}, vote_type: {vote_type}")
            _result = _get_result_csc(
                gaps,
                class_sizes_gt=class_sizes,
                anomaly_labels=anomaly_labels,
                vote_type=vote_type,
                th_type=th_type,
            )
            _result["vote_type"] = vote_type
            _result["th_type"] = th_type
            _result["method"] = "csc"

            results.append(_result)

    df = utils.save_dicts_to_csv(results, filename=os.path.join(save_dir, "result.csv"))

    utils.print_df(df)

    if plot_self_sim:
        self_sim = class_size.compute_self_sim(gaps)
        _anomaly_labels = anomaly_labels.to(torch.bool).tolist()
        _few_shot_labels = (class_sizes <= 20).tolist()

        self_sim_abnormal_plot_dir = os.path.join(save_dir, "self_sim_abnormal")
        self_sim_few_shot_plot_dir = os.path.join(save_dir, "self_sim_few_shot")
        self_sim_else_plot_dir = os.path.join(save_dir, "self_sim_else")
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
                anomaly_labels,
                th=ths["random-ideal"],
                filename=_filename,
            )

        Parallel(n_jobs=-1)(delayed(plot_gap_self_sim)(i) for i in range(len(self_sim)))


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


def convert_indices_to_bool(n, indices: torch.Tensor):
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
