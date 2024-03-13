"""
For research only
"""

import os
import torch
import math
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import torch.nn as nn

from collections import Counter
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import wasserstein_distance

import src.evaluator.result as result
import src.class_size as class_size
import src.adaptive_class_size as adaptive_class_size
import src.helpers.cv2plot as cv2plot

from src import utils
from src.dataloader import get_dataloaders
from src.get_args import parse_args  # FIXME: make independent args
from src.engine import AblationEngine
from src.backbone import get_backbone
from src.feature_embedder import FeatureEmbedder

from src.patch_maker import PatchMaker
from src.sampler import LOFSampler, TailSampler, TailedLOFSampler, AdaptiveTailSampler


def analyze_gap(args):

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

    train_dataloader = dataloaders[0]["train"]

    backbone_name = config.model.backbone_names[0]

    save_train_dir_path = os.path.join(
        "./artifacts", args.data_name, args.config_name, train_dataloader.name
    )

    ablation_engine = AblationEngine(
        config=config,
        backbone_name=backbone_name,
        device=device,
        input_shape=input_shape,
        train_dataloader=train_dataloader,
        test_dataloader=None,
        faiss_on_gpu=args.faiss_on_gpu,
        faiss_num_workers=args.faiss_num_workers,
        sampler_on_gpu=args.sampler_on_gpu,
        save_dir_path=save_train_dir_path,
        patch_infer=args.patch_infer,
        train_mode=getattr(config.model, "train_mode", None),
    )

    embedding_extractor = ablation_engine.set_embedding_extractor(iter=config.model.embedding_extractor_iter)
    embedding_extractor.fc = nn.Identity()
    embedding_extractor.eval()

    _, names_to_ints = ablation_engine._get_dataset_info()

    labels = [[]]*len(train_dataloader)
    gaps = [[]]*len(train_dataloader)
    is_anomaly = [[]]*len(train_dataloader)

    for i, data in enumerate(train_dataloader):

        images = data["image"].to(device)
        label_names = data["classname"]
        image_names = data["image_name"]
        file_names = [os.path.splitext(os.path.basename(image_name))[0] for image_name in image_names]
        _is_anomaly = torch.tensor([not filename.isnumeric() for filename in file_names], dtype=torch.bool)
        _labels = names_to_ints(label_names)
        with torch.no_grad():
            _gaps = embedding_extractor(images.to(device)).cpu()

        labels[i] = _labels
        gaps[i] = _gaps
        is_anomaly[i] = _is_anomaly
    
    gaps = torch.cat(gaps, dim=0)
    labels = torch.cat(labels, dim=0)
    is_anomaly = torch.cat(is_anomaly, dim=0)

    unique_labels, counts = labels.unique(return_counts=True)
    class_sizes_dict = {label.item(): count.item() for label, count in zip(unique_labels, counts)}

    class_sizes = torch.tensor([class_sizes_dict[label.item()] for label in labels], dtype=torch.long)

    save_dir = os.path.join('.', 'logs', args.data_name, args.config_name)

    _evaluate_tail_class_detection(
        gaps=gaps,
        class_sizes_gt=class_sizes,
        is_anomaly_gt=is_anomaly,
        save_dir=save_dir,
    )
def _evaluate_tail_class_detection(
    gaps: torch.Tensor,
    class_sizes_gt: torch.Tensor,
    is_anomaly_gt: torch.Tensor,
    save_dir: str,
):

    method_names = [
        "acs-trim_min-mode",
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

    return df


def _get_result_tail_class_detection(
    method_name,
    class_sizes_gt,
    is_anomaly_gt,
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
    elif method_name == "lof_norm":
        gaps = F.normalize(gaps, dim=-1)
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

    elif method_name == "if_norm":
        from sklearn.ensemble import IsolationForest
        gaps = F.normalize(gaps, dim=-1)
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
    elif method_name == "ocsvm_norm":
        from sklearn.svm import OneClassSVM
        gaps = F.normalize(gaps, dim=-1)
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
    elif method_name == "dbscan_norm":
        from sklearn.cluster import DBSCAN
        gaps = F.normalize(gaps, dim=-1)
        X = gaps.numpy()
        dbscan = DBSCAN().fit(X)
        labels = dbscan.labels_
        is_tail_pred = torch.from_numpy(labels == -1).long()
        tail_scores = is_tail_pred.float()
        cluster_counts = Counter(labels)
        class_sizes_pred = torch.FloatTensor(
            np.array([cluster_counts[label] for label in labels])
        )
    
    elif method_name == "dbscan_elbow":
        from sklearn.cluster import DBSCAN

        X = gaps.numpy()
        dbscan = DBSCAN(min_samples=1).fit(X)
        labels = dbscan.labels_
        cluster_counts = Counter(labels)
        class_sizes_pred = torch.FloatTensor(
            np.array([cluster_counts[label] for label in labels])
        )
        tail_scores = 1 - class_sizes_pred / class_sizes_pred.max()
        is_tail_pred = class_size.predict_few_shot_class_samples(class_sizes_pred)
    elif method_name == "dbscan_elbow_norm":
        from sklearn.cluster import DBSCAN

        X = gaps.numpy()
        dbscan = DBSCAN(min_samples=1).fit(X)
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
    elif method_name == "kmeans_norm":
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
    elif method_name == "gmm_norm":
        from sklearn.mixture import GaussianMixture
        gaps = F.normalize(gaps, dim=-1)
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
        X = gaps.numpy()
        kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(X)
        log_densities = kde.score_samples(X)
        log_densities = torch.from_numpy(log_densities).float()
        tail_scores = -log_densities
        class_sizes_pred = torch.zeros_like(tail_scores)
        is_tail_pred = (log_densities < class_size.elbow(log_densities, quantize=True)).long()
    
    elif method_name == "kde_norm":
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
    elif method_name == "affprop_norm":  # affinity propagation
        from sklearn.cluster import AffinityPropagation
        gaps = F.normalize(gaps, dim=-1)
        X = gaps.numpy()
        affinity_propagation = AffinityPropagation(random_state=0).fit(X)
        labels = affinity_propagation.predict(X)
        cluster_counts = Counter(labels)
        class_sizes_pred = torch.from_numpy(
            np.array([cluster_counts[label] for label in labels])
        ).float()
        tail_scores = 1 - class_sizes_pred / class_sizes_pred.max()
        is_tail_pred = class_size.predict_few_shot_class_samples(class_sizes_pred)
    elif method_name == "knnod":
        from pyod.models.knn import KNN
        X = gaps.numpy()
        knn = KNN()
        knn.fit(X)
        is_tail_pred = torch.from_numpy(knn.labels_)  # binary labels (0: inliers, 1: outliers)
        tail_scores = torch.from_numpy(knn.decision_scores_)  # raw outlier scores
        class_sizes_pred = torch.zeros_like(tail_scores)
    elif method_name == "knnod_norm":
        from pyod.models.knn import KNN
        gaps = F.normalize(gaps, dim=-1)
        X = gaps.numpy()
        knn = KNN()
        knn.fit(X)
        is_tail_pred = torch.from_numpy(knn.labels_)  # binary labels (0: inliers, 1: outliers)
        tail_scores = torch.from_numpy(knn.decision_scores_)  # raw outlier scores
        class_sizes_pred = torch.zeros_like(tail_scores)
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

    precision, recall, thresholds = metrics.precision_recall_curve(is_tail_gt, tail_scores)
    auprc = metrics.auc(recall, precision)

    return {
        "method": method_name,
        "ratio_missing_tail": ratio_missing_tail * 100,
        "ratio_included_anomaly": ratio_included_anomaly * 100,
        "ratio_included_head": ratio_included_head * 100,
        "auprc": auprc * 100,
        "class_size_pred_error": class_size_pred_error,
        "auroc": metrics.roc_auc_score(is_tail_gt, tail_scores) * 100,
        "tail_pred_acc": tail_pred_acc * 100,
        "tail_pred_mcc": metrics.matthews_corrcoef(is_tail_gt, is_tail_pred),
        "tail_pred_precision": metrics.precision_score(is_tail_gt, is_tail_pred),
        "tail_pred_recall": metrics.recall_score(is_tail_gt, is_tail_pred),
    }

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

# mvtec:
if __name__ == "__main__":

    args = parse_args()
    analyze_gap(args)
    
