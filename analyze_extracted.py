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
import src.adaptive_class_size as adaptive_class_size
import src.helpers.cv2plot as cv2plot

from src import utils
from src.dataloader import get_dataloaders
from src.get_args import parse_args  # FIXME: make independent args
from src.engine import Engine
from src.backbone import get_backbone
from src.feature_embedder import FeatureEmbedder

from src.patch_maker import PatchMaker
from src.sampler import LOFSampler, TailSampler, TailedLOFSampler, AdaptiveTailSampler


def analyze_patch(extracted_path: str, data_name: str, config_name: str):

    assert os.path.exists(extracted_path)

    extracted = torch.load(extracted_path)

    feas = extracted["feas"]
    masks = extracted["masks"]

    gaps = extracted["gaps"]
    class_names = extracted["class_names"]
    class_sizes = extracted["class_sizes"]

    num_samples_per_class = dict(Counter(class_names))

    save_log_dir = os.path.join("./logs", f"{data_name}_{config_name}")

    save_data_info_path = os.path.join(save_log_dir, "num_samples_per_class.csv")
    utils.save_dicts_to_csv([num_samples_per_class], save_data_info_path)

    num_classes = len(set(class_names))

    df = get_patch_result_df(
        feas,
        masks,
        save_log_dir,
    )
    return df


def get_patch_result_df(feas: torch.Tensor, masks, save_dir):
    downsized_masks = _downsize_masks(masks, mode="bilinear")

    if downsized_masks.ndim == 4:
        downsized_masks = downsized_masks[:, 0, :, :]

    n, fea_dim, h, w = feas.shape

    anomaly_patch_scores_gt = downsized_masks.reshape((-1))
    is_anomaly_patch_gt = torch.round(anomaly_patch_scores_gt).to(torch.long)
    feature_map_shape = [28, 28]
    features = (
        feas.reshape((feas.shape[0], feas.shape[1], -1))
        .permute(0, 2, 1)
        .reshape((-1, fea_dim))
    )

    return _evaluate_anomaly_patch_detection(
        is_anomaly_patch_gt, features, feature_map_shape, save_dir
    )


def _evaluate_anomaly_patch_detection(
    is_anomaly_patch_gt, features, feature_map_shape, save_dir
) -> dict:

    method_names = ["lof", "lof-scs_symmin", "scs_symmin"]
    results = []
    for method_name in method_names:
        _result = _get_result_anomaly_patch_detection(
            method_name,
            is_anomaly_patch_gt,
            features,
            feature_map_shape,
        )
        results.append(_result)

    df = utils.save_dicts_to_csv(
        results, filename=os.path.join(save_dir, "result_patch.csv")
    )
    utils.print_df(df)

    return df


def _get_result_anomaly_patch_detection(
    method_name,
    is_anomaly_patch_gt,
    features,
    feature_map_shape,
):

    if method_name == "lof":
        _, lof_idxes, outlier_scores = LOFSampler().run(
            features, feature_map_shape, return_outlier_scores=True
        )
        is_anomaly_patch_pred = 1 - convert_indices_to_bool(len(features), lof_idxes)
    elif method_name == "lof-scs_symmin":
        _, lofcsp_idxes, outlier_scores = TailedLOFSampler(tail_th_type="symmin").run(
            features, feature_map_shape, return_outlier_scores=True
        )
        is_anomaly_patch_pred = 1 - convert_indices_to_bool(len(features), lofcsp_idxes)
    elif method_name == "scs_symmin":
        _, lofcsp_idxes, outlier_scores = TailedLOFSampler(
            tail_th_type="symmin", without_lof=True
        ).run(features, feature_map_shape, return_outlier_scores=True)
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
        "auroc": round(
            metrics.roc_auc_score(is_anomaly_patch_gt, outlier_scores) * 100, 2
        ),
    }

    return _result


def analyze_gap(extracted_path: str, data_name: str, config_name: str):

    assert os.path.exists(extracted_path)

    extracted = torch.load(extracted_path)

    masks = extracted["masks"]

    gaps = extracted["gaps"]
    class_names = extracted["class_names"]
    class_sizes = extracted["class_sizes"]

    num_samples_per_class = dict(Counter(class_names))

    save_log_dir = os.path.join("./logs", f"{data_name}_{config_name}")

    save_data_info_path = os.path.join(save_log_dir, "num_samples_per_class.csv")
    utils.save_dicts_to_csv([num_samples_per_class], save_data_info_path)

    num_classes = len(set(class_names))
    # save_plot_dir = os.path.join("./artifacts", data_name, config_name)
    # print(f"save_plot_dir: {save_plot_dir}")
    # plot_gap_analysis(gaps, masks, class_names, class_sizes, num_classes, save_plot_dir)
    # return
    df = get_gap_result_df(
        gaps,
        masks,
        class_names,
        class_sizes,
        num_classes,
        save_log_dir,
    )
    
    return df


def get_gap_result_df(gaps, masks, class_names, class_sizes, num_classes, save_dir):
    if gaps.ndim == 4:
        gaps = gaps[:, :, 0, 0]

    class_labels, class_label_names = _convert_class_names_to_labels(class_names)
    class_labels = class_labels.numpy()

    is_anomaly_gt = (masks.sum(dim=(1, 2, 3)) > 0).to(torch.long)

    return _evaluate_tail_class_detection(
        gaps=gaps,
        class_sizes_gt=class_sizes,
        is_anomaly_gt=is_anomaly_gt,
        num_classes=num_classes,
        save_dir=save_dir,
    )


def _evaluate_tail_class_detection(
    gaps: torch.Tensor,
    class_sizes_gt: torch.Tensor,
    is_anomaly_gt: torch.Tensor,
    num_classes: int,
    save_dir: str,
):

    method_names = [
        "acs-max_step-none",
        "acs-max_step-mode",
        "acs-double_max_step-none",
        "acs-double_max_step-mode",
        "acs-double_min_bin_count-none",
        "acs-double_min_bin_count-mode",
        "acs-min_kde-none",
        "acs-double_min_kde-none",
        "scs_symmin",
        "scs_indep",
        "lof",
        "if",
        "ocsvm",
        "dbscan",
        "dbscan_adaptive",
        "dbscan_adaptive_elbow",
        "kmeans",
        "gmm",
        "kde",
        "affprop",
    ]

    results = []
    for method_name in method_names:
        _result = _get_result_tail_class_detection(
            method_name=method_name,
            class_sizes_gt=class_sizes_gt,
            is_anomaly_gt=is_anomaly_gt,
            num_classes=num_classes,
            gaps=gaps,
        )
        results.append(_result)

    df = utils.save_dicts_to_csv(
        results, filename=os.path.join(save_dir, "result_gap.csv")
    )

    utils.print_df(df)

    return df


def _get_result_tail_class_detection(
    method_name,
    class_sizes_gt,
    is_anomaly_gt,
    num_classes,
    gaps: torch.Tensor,
):

    if "acs" in method_name:
        self_sim = class_size.compute_self_sim(gaps)
        method_parts = method_name.split("-")
        th_type = method_parts[1]
        vote_type = method_parts[2]
        tail_sampler = AdaptiveTailSampler(th_type=th_type, vote_type=vote_type)
        _, tail_indices, class_sizes_pred = tail_sampler.run(
            gaps, return_class_sizes=True
        )
        
        num_samples_per_class = class_size.predict_num_samples_per_class(class_sizes_pred)
        max_K = class_size.predict_max_K(num_samples_per_class)

        is_tail_pred = convert_indices_to_bool(len(gaps), tail_indices)
        tail_scores = 1 - class_sizes_pred / class_sizes_pred.max()
    elif "scs" in method_name:
        method_parts = method_name.split("_")
        th_type = method_parts[1]
        tail_sampler = TailSampler(th_type=th_type, vote_type="mean")
        _, tail_indices, class_sizes_pred = tail_sampler.run(
            gaps, return_class_sizes=True
        )
        
        # TODO: observer
        num_samples_per_class = class_size.predict_num_samples_per_class(class_sizes_pred)
        max_K = class_size.predict_max_K(num_samples_per_class)

        is_tail_pred = convert_indices_to_bool(len(gaps), tail_indices)
        tail_scores = 1 - class_sizes_pred / class_sizes_pred.max()
    elif method_name == "lof":
        lof_sampler = LOFSampler()
        _, head_indices, tail_scores = lof_sampler.run(gaps, return_outlier_scores=True)
        is_head_pred = convert_indices_to_bool(len(gaps), head_indices)
        is_tail_pred = 1 - is_head_pred
        class_sizes_pred = torch.zeros((len(gaps)))
    elif method_name == "if":
        from sklearn.ensemble import IsolationForest

        clf = IsolationForest(random_state=0)
        X = gaps.numpy()
        clf.fit(X)
        tail_scores = -torch.from_numpy(clf.decision_function(X)).float()
        is_tail_pred = torch.from_numpy(clf.predict(X) == -1).long()
        class_sizes_pred = torch.zeros_like(tail_scores)
    elif method_name == "ocsvm":
        from sklearn.svm import OneClassSVM

        X = gaps.numpy()
        ocsvm = OneClassSVM(gamma="auto").fit(X)
        tail_scores = -torch.from_numpy(ocsvm.decision_function(X)).float()
        is_tail_pred = torch.from_numpy(ocsvm.predict(X) == -1).long()
        class_sizes_pred = torch.zeros_like(tail_scores)

    elif method_name == "dbscan":
        from sklearn.cluster import DBSCAN

        X = gaps.numpy()
        dbscan = DBSCAN().fit(X)
        labels = dbscan.labels_
        is_tail_pred = torch.from_numpy(labels == -1).long()
        tail_scores = is_tail_pred.float()
        cluster_counts = Counter(labels)
        class_sizes_pred = torch.FloatTensor(
            np.array([cluster_counts[label] for label in labels])
        )
    
    elif method_name == "dbscan_adaptive":
        from sklearn.cluster import DBSCAN

        self_sim = class_size.compute_self_sim(gaps, normalize=True)
        min_sim = max(class_size.compute_self_sim_min(self_sim).item(), 0)
        X = F.normalize(gaps, dim=-1).numpy()
        dbscan = DBSCAN(min_samples=1, eps=np.cos(np.arccos(min_sim) / 2)).fit(X)
        labels = dbscan.labels_
        is_tail_pred = torch.from_numpy(labels == -1).long()
        tail_scores = is_tail_pred.float()
        cluster_counts = Counter(labels)
        class_sizes_pred = torch.FloatTensor(
            np.array([cluster_counts[label] for label in labels])
        )
    elif method_name == "dbscan_adaptive_elbow":
        from sklearn.cluster import DBSCAN

        self_sim = class_size.compute_self_sim(gaps, normalize=True)
        min_sim = max(class_size.compute_self_sim_min(self_sim).item(), 0)
        X = F.normalize(gaps, dim=-1).numpy()
        dbscan = DBSCAN(min_samples=1, eps=np.cos(np.arccos(min_sim) / 2)).fit(X)
        labels = dbscan.labels_
        cluster_counts = Counter(labels)
        class_sizes_pred = torch.FloatTensor(
            np.array([cluster_counts[label] for label in labels])
        )
        tail_scores = 1 - class_sizes_pred / class_sizes_pred.max()
        is_tail_pred = class_size.predict_few_shot_class_samples(class_sizes_pred)
    elif method_name == "kmeans":
        from sklearn.cluster import KMeans

        X = gaps.numpy()
        kmeans = KMeans(n_clusters=3, random_state=0).fit(
            X
        )  # Adjust 'n_clusters' as needed
        labels = kmeans.predict(X)
        cluster_counts = Counter(labels)
        class_sizes_pred = torch.from_numpy(
            np.array([cluster_counts[label] for label in labels])
        ).float()
        tail_scores = 1 - class_sizes_pred / class_sizes_pred.max()
        is_tail_pred = class_size.predict_few_shot_class_samples(class_sizes_pred, percentile=1)
    elif method_name == "gmm":
        from sklearn.mixture import GaussianMixture

        X = gaps.numpy()
        gmm = GaussianMixture().fit(X)
        labels = gmm.predict(X)
        cluster_counts = Counter(labels)
        class_sizes_pred = torch.from_numpy(
            np.array([cluster_counts[label] for label in labels])
        ).float()
        tail_scores = 1 - class_sizes_pred / class_sizes_pred.max()
        is_tail_pred = class_size.predict_few_shot_class_samples(class_sizes_pred, percentile=1)
    elif method_name == "kde":
        from sklearn.neighbors import KernelDensity
        gaps = F.normalize(gaps, dim=-1)
        X = gaps.numpy()
        kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(X)
        log_densities = kde.score_samples(X)
        log_densities = torch.from_numpy(log_densities).float()
        tail_scores = -log_densities
        class_sizes_pred = torch.zeros_like(tail_scores)
        is_tail_pred = (log_densities < class_size.elbow(log_densities, quantize=True)).long()

    elif method_name == "affprop":  # affinity propagation
        from sklearn.cluster import AffinityPropagation

        X = gaps.numpy()
        affinity_propagation = AffinityPropagation(random_state=0).fit(X)
        labels = affinity_propagation.predict(X)
        cluster_counts = Counter(labels)
        class_sizes_pred = torch.from_numpy(
            np.array([cluster_counts[label] for label in labels])
        ).float()
        tail_scores = 1 - class_sizes_pred / class_sizes_pred.max()
        is_tail_pred = class_size.predict_few_shot_class_samples(class_sizes_pred)
    else:
        raise NotImplementedError()

    is_tail_gt = (class_sizes_gt <= 20).to(torch.long)

    is_missing_tail = (is_tail_pred - is_tail_gt) < 0
    is_included_anomaly = (is_tail_pred + is_anomaly_gt) > 1
    is_included_head = (is_tail_pred - is_tail_gt) > 0

    ratio_missing_tail = is_missing_tail.sum().item() / is_tail_gt.sum().item()
    ratio_included_anomaly = (
        is_included_anomaly.sum().item() / is_anomaly_gt.sum().item()
    )
    ratio_included_head = is_included_head.sum().item() / max(
        is_tail_pred.sum().item(), 1
    )

    class_size_pred_error = round(
        (abs(class_sizes_pred - class_sizes_gt) * (1 / class_sizes_gt)).mean().item(), 2
    )
    tail_pred_acc = 1 - abs(is_tail_gt - is_tail_pred).to(torch.float).mean().item()

    return {
        "method": method_name,
        "ratio_missing_tail": ratio_missing_tail * 100,
        "ratio_included_anomaly": ratio_included_anomaly * 100,
        "ratio_included_head": ratio_included_head * 100,
        "auroc": metrics.roc_auc_score(is_tail_gt, tail_scores) * 100,
        "class_size_pred_error": class_size_pred_error,
        "tail_pred_acc": tail_pred_acc * 100,
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


def plot_patch_analysis(feas: torch.Tensor, masks, save_dir, save_plot=True):

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


def plot_gap_analysis(
    gaps, masks, class_names, class_sizes, num_classes, save_dir
):

    if gaps.ndim == 4:
        gaps = gaps[:, :, 0, 0]

    class_labels, class_label_names = _convert_class_names_to_labels(class_names)
    class_labels = class_labels.numpy()

    is_anomaly_gt = (masks.sum(dim=(1, 2, 3)) > 0).to(torch.long)

    _anomaly_labels = is_anomaly_gt.to(torch.bool).tolist()
    _few_shot_labels = (class_sizes < 20).tolist()
    save_plot_dir = os.path.join(save_dir, "plot_gap")

    self_sim = class_size.compute_self_sim(gaps)
    class_sizes_pred = adaptive_class_size.predict_adaptive_class_sizes(self_sim)

    self_sim_abnormal_plot_dir = os.path.join(save_plot_dir, "self_sim_abnormal")
    self_sim_few_shot_plot_dir = os.path.join(save_plot_dir, "self_sim_few_shot")
    self_sim_else_plot_dir = os.path.join(save_plot_dir, "self_sim_else")
    os.makedirs(self_sim_abnormal_plot_dir, exist_ok=True)
    os.makedirs(self_sim_few_shot_plot_dir, exist_ok=True)
    os.makedirs(self_sim_else_plot_dir, exist_ok=True)

    ths = adaptive_class_size._compute_ths(self_sim, "double_max_step")

    def plot_gap_self_sim(index):
        # print(f"plotting ngap for {index}")
        _scores = self_sim[index].numpy()
        _is_anomaly = _anomaly_labels[index]
        _is_few_shot = _few_shot_labels[index]
        if _is_anomaly:
            _filename = os.path.join(self_sim_abnormal_plot_dir, f"{index:04d}-{class_names[index]}.jpg")
        elif _is_few_shot:
            _filename = os.path.join(self_sim_few_shot_plot_dir, f"{index:04d}-{class_names[index]}.jpg")
        else:
            _filename = os.path.join(self_sim_else_plot_dir, f"{index:04d}-{class_names[index]}.jpg")
        plot_scatter(
            _scores,
            class_labels,
            class_label_names,
            is_anomaly_gt,
            th=ths[index].item(),
            filename=_filename,
        )

    Parallel(n_jobs=-1)(delayed(plot_gap_self_sim)(i) for i in range(len(self_sim)))

    # for i in range(len(self_sim)):
    #     plot_gap_self_sim(i)


import pandas as pd
from typing import List


import pandas as pd
import numpy as np
from typing import List


def average_dfs(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    combined_df = pd.concat(dfs)

    # Identify numeric columns
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns

    # Compute the mean and std only for numeric columns
    avg_numeric_df = combined_df[numeric_cols].groupby(combined_df.index).mean()
    std_numeric_df = combined_df[numeric_cols].groupby(combined_df.index).std()

    # Rename columns in std_numeric_df to indicate they are standard deviations
    std_numeric_df = std_numeric_df.add_suffix("_std")

    # Extract non-numeric columns from the first dataframe in the list
    non_numeric_df = dfs[0][combined_df.select_dtypes(exclude=[np.number]).columns]

    # Concatenate non-numeric, average, and std dataframes
    avg_df = pd.concat([non_numeric_df, avg_numeric_df, std_numeric_df], axis=1)

    return avg_df


def analyze(data="mvtec_all", type="gap", seeds: list=list(range(101,106))):

    data_names = get_data_names(data, seeds=seeds,)

    config_names = [
        "extract_mvtec_01",
        # "extract_mvtec_02",
        # "extract_mvtec_03",
        # "extract_mvtec_04",
    ]

    dfs = []

    for config_name in config_names:
        for data_name in data_names:
            print(f"config: {config_name} data: {data_name}")
            extracted_path = f"./artifacts/{data_name}_mvtec-multiclass/{config_name}/extracted_train_all.pt"

            if type == "gap":
                _df = analyze_gap(
                    extracted_path=extracted_path,
                    data_name=data_name,
                    config_name=config_name,
                )
            elif type == "patch":
                _df = analyze_patch(
                    extracted_path=extracted_path,
                    data_name=data_name,
                    config_name=config_name,
                )
            else:
                raise NotImplementedError()
            dfs.append(_df)

    avg_df = average_dfs(dfs)
    os.makedirs("./logs", exist_ok=True)
    avg_df.to_csv(f"./logs/analysis-{data}-{type}.csv", index=False)


def get_data_names(data: str, seeds: list):

    mvtec_data_base_names = [
        "mvtec_step_random_nr10_tk1_tr60",
        "mvtec_step_random_nr10_tk4_tr60",
        "mvtec_pareto_random_nr10",
    ]

    visa_data_base_names = [
        "visa_step_random_nr05_tk1_tr60",
        "visa_step_random_nr05_tk4_tr60",
        "visa_pareto_random_nr05",
    ]

    if data == "mvtec_all":
        data_base_names = mvtec_data_base_names
    elif data == "mvtec_step_tk1":
        data_base_names = [mvtec_data_base_names[0]]
    elif data == "mvtec_step_tk4":
        data_base_names = [mvtec_data_base_names[1]]
    elif data == "mvtec_step_pareto":
        data_base_names = [mvtec_data_base_names[2]]
    elif data == "":
        data_base_names = [mvtec_data_base_names[2]]
    elif data == "visa_all":
        data_base_names = visa_data_base_names
    elif data == "visa_step_tk1":
        data_base_names = [visa_data_base_names[0]]
    elif data == "visa_step_tk4":
        data_base_names = [visa_data_base_names[1]]
    elif data == "visa_step_pareto":
        data_base_names = [visa_data_base_names[2]]
    elif data == "":
        data_base_names = [visa_data_base_names[2]]
    else:
        raise NotImplementedError()

    data_names = [
        f"{data_base_name}_seed{seed}"
        for data_base_name in data_base_names
        for seed in seeds
    ]
    
    return data_names


# mvtec:
if __name__ == "__main__":
    utils.set_seed(0)
    seeds_visa = [200, 201, 202, 203, 204]
    seeds_mvtec = [101, 102, 103, 104, 105]
    analyze(data="mvtec_all", type="gap", seeds=seeds_mvtec)
    analyze(data="visa_all", type="gap", seeds=seeds_visa)
    # analyze(data="mvtec_step_tk4", type="gap", seeds=seeds_mvtec)
    # analyze(data="mvtec_step_tk1", type="gap", seeds=seeds_mvtec)   
    # analyze(data="mvtec_step_pareto", type="gap", seeds=seeds_mvtec)
    # analyze(data="mvtec_step_tk4", type="patch")
    # analyze(data="mvtec_step_tk1", type="patch")
    # analyze(data="mvtec_step_pareto", type="patch")
    
