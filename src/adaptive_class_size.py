import torch
import math
import gc
import numpy as np
import torch.nn.functional as F
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import mode

from . import utils
from . import class_size


def predict_adaptive_class_sizes(
    self_sim: torch.Tensor, 
    th_type: str = "double_max_step",
    vote_type: str = "none",
):

    ths = _compute_ths(self_sim, th_type)

    mask = self_sim >= ths
    count_map = mask * mask.sum(dim=1, keepdim=True)

    class_sizes = _vote(count_map, vote_type)
    
    return class_sizes

def _vote(count_map: torch.Tensor, vote_type) -> torch.Tensor:
    
    if vote_type == "none":
        return _vote_none(count_map)
    elif vote_type == "mode":
        return _vote_mode(count_map)
    elif vote_type == "mean":
        return _vote_mean(count_map)

def _vote_mean(count_map):
    class_sizes = count_map.sum(dim=0) / torch.count_nonzero(count_map, dim=0)
    return class_sizes.to(torch.float)

def _vote_none(count_map: torch.Tensor):
    class_sizes = count_map.float().mean(dim=1)
    return class_sizes.to(torch.float)

def _vote_mode(count_map: torch.Tensor):
    count_map = count_map.numpy()
    count_map_naned = np.where(count_map == 0, np.nan, count_map)
    class_sizes = mode(
        count_map_naned, axis=0, nan_policy="omit"
    ).mode
    return torch.from_numpy(class_sizes).to(torch.float)

def _compute_ths(self_sim: torch.Tensor, th_type) -> torch.Tensor:
    self_sim = self_sim.numpy()

    if th_type == "max_step":
        _compute_th = _compute_th_max_step
    elif th_type == "max_step_min_num_neighbors":
        _compute_th = _compute_th_max_step_min_num_neighbors
    elif th_type == "double_max_step":
        _compute_th = _compute_th_double_max_step

    n = len(self_sim)
    ths = [[]] * n

    def compute_th(i):
        _sim = self_sim[i]
        _th = _compute_th(_sim)
        return _th

    # for i in range(n):
    #     ths[i] = compute_th(i)

    ths = Parallel(n_jobs=-1)(delayed(compute_th)(i) for i in range(n))

    ths = torch.FloatTensor(ths)[:, None]
    return ths


def _compute_th_max_step(scores: np.ndarray):
    scores_sorted = _sort(scores, descending=False)
    diff = np.diff(scores_sorted)

    idx_max_step = diff.argmax()

    th = (scores_sorted[idx_max_step + 1] + scores_sorted[idx_max_step]) / 2

    return th


def _compute_th_double_max_step(scores: np.ndarray):
    scores_sorted = _sort(scores, descending=False)
    diff = np.diff(scores_sorted)

    crit1 = diff

    crit2 = np.empty_like(crit1)
    for i in range(len(crit2) - 1):
        crit2[i] = np.max(diff[i + 1 :])

    crit2[-1] = diff[-1]

    crit = crit1 / np.maximum(crit2, 1e-7)

    idx_max_step = crit.argmax()

    th = (scores_sorted[idx_max_step] + scores_sorted[idx_max_step + 1]) / 2

    return th


def _compute_th_max_step_min_num_neighbors(scores: np.ndarray):
    scores_sorted = _sort(scores, descending=True)
    diff = -np.diff(scores_sorted)

    crit1 = diff
    crit2 = -np.log(np.arange(len(diff)) + 1) + math.log(len(diff))

    crit = crit1 * crit2

    idx_max_step = crit.argmax()

    th = (scores_sorted[idx_max_step] + scores_sorted[idx_max_step + 1]) / 2

    return th


def _sort(scores: np.ndarray, descending: bool = False, quantize=True) -> np.ndarray:

    dtype = scores.dtype

    if quantize:
        n = len(scores)
        scores = (n * scores).astype(np.int_)
    else:
        n = 1

    scores = np.sort(scores).astype(dtype) / n

    if descending:
        scores = scores[::-1]

    return scores


def adaptively_sample_few_shot(
    X: torch.Tensor,
    th_type: str = "max_step_min_num_neighbors",
    vote_type: str = "none",
    return_class_sizes=False,
):
    self_sim = class_size.compute_self_sim(X)
    class_sizes = predict_adaptive_class_sizes(self_sim, th_type, vote_type)
    if return_class_sizes:
        return class_sizes

    is_few_shot = class_size.predict_few_shot_class_samples(class_sizes)

    sample_indices = torch.where(is_few_shot == 1)[0]

    return X[sample_indices], sample_indices
