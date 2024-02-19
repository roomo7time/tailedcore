import torch
import time
import numpy as np
import torch.nn.functional as F
from joblib import Parallel, delayed
from tqdm import tqdm
from . import utils

from scipy.stats import mode

def compute_self_sim(X: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    if normalize:
        X = F.normalize(X, dim=-1)

    if X.ndim == 2:
        Xt = X.t()
    elif X.ndim == 3:
        Xt = X.permute(0, 2, 1)
    else:
        raise NotImplementedError()

    self_sim = torch.matmul(X, Xt)

    return self_sim


def predict_class_sizes(self_sim: torch.Tensor, th, vote_type="mean") -> torch.Tensor:
    b = self_sim.shape[0]
    mask = self_sim >= th

    count_map = mask * mask.sum(dim=1, keepdim=True)

    if vote_type == "mean":
        class_sizes = count_map.sum(dim=0) / torch.count_nonzero(count_map, dim=0)
    elif vote_type == "mode":
        count_map = count_map.numpy()
        count_map_naned = np.where(count_map == 0, np.nan, count_map)
        class_sizes = mode(
            count_map_naned, axis=0, nan_policy="omit"
        ).mode  # FIXME: what happens when there is no majority?
    elif vote_type == "nearest":
        columnwise_means = (
            count_map.sum(dim=0, keepdim=True)
            / torch.count_nonzero(count_map, dim=0)[None, :]
        )
        idxes = torch.argmin((count_map - columnwise_means).abs(), dim=0)
        class_sizes = count_map[torch.arange(b), idxes]

    if isinstance(class_sizes, np.ndarray):
        class_sizes = torch.from_numpy(class_sizes)

    return class_sizes.to(torch.float)



def predict_num_samples_per_class(class_sizes: torch.FloatTensor, round_class_sizes=True) -> torch.FloatTensor:
    # FIXME: topk average is required for robustness
    if round_class_sizes:
        class_sizes = torch.round(class_sizes).to(torch.long)
    _class_sizes_sorted = torch.sort(class_sizes, descending=True)[0]
    class_sizes_sorted = torch.maximum(_class_sizes_sorted, torch.ones_like(_class_sizes_sorted))

    num_samples_per_class = []

    while len(class_sizes_sorted) > 0:
        _num_samples_in_current_class = min(
            class_sizes_sorted[0], len(class_sizes_sorted)
        )
        _num_samples_in_current_class = torch.round(class_sizes_sorted[:_num_samples_in_current_class].float().mean()).long().item()
        _num_samples_in_current_class = min(_num_samples_in_current_class, len(class_sizes_sorted))
        num_samples_per_class.append(_num_samples_in_current_class)
        class_sizes_sorted = class_sizes_sorted[_num_samples_in_current_class:]

    return torch.FloatTensor(num_samples_per_class)


# FIXME: this function is not robust. needs to be revised
def predict_max_few_shot_class_size(num_samples_per_class: torch.FloatTensor) -> float:
    
    idx = detect_max_step(num_samples_per_class, round_arr=False)
    
    return num_samples_per_class[idx].item()


def predict_few_shot_class_samples(class_sizes: torch.Tensor, wide_cover: bool = True) -> torch.Tensor:

    num_samples_per_class = predict_num_samples_per_class(class_sizes)

    _num_samples_per_class = torch.cat([torch.arange(len(num_samples_per_class))[:, None], num_samples_per_class[:, None]], dim=1)
    ods = compute_orthogonal_distances(_num_samples_per_class, _num_samples_per_class[[0, -1], :])
    max_K_idx = ods.argmax() + 1
    max_K = num_samples_per_class[max_K_idx].item()
    
    few_shot_idxes = (class_sizes <= max_K).to(torch.long)
    return few_shot_idxes

def _detect_max_step_idx(arr: torch.FloatTensor, quantize=True, factor=1.):
    if quantize:
        # arr = factor*torch.maximum(torch.round(arr), torch.ones_like(arr))
        arr = torch.round(arr*factor)
        arr = arr.to(torch.long)
    sorted_arr = torch.sort(arr, descending=True)[0]
    # utils.plot_scores(sorted_arr, markersize=2, alpha=1)
    sorted_arr_shifted = torch.empty_like(sorted_arr)
    sorted_arr_shifted[0:-1] = sorted_arr[1:]
    sorted_arr_shifted[-1] = sorted_arr[-1]

    factors = sorted_arr / sorted_arr_shifted
    idx = min(int(factors.argmax()) + 1, len(sorted_arr) - 1)

    return idx

def detect_max_step(arr: torch.FloatTensor, quantize=True, factor=1.):
    if quantize:
        # arr = factor*torch.maximum(torch.round(arr), torch.ones_like(arr))
        _arr = torch.round(arr*factor)
        _arr = _arr.to(torch.long)
    sorted_arr = torch.sort(_arr, descending=True)[0]
    
    _line = [torch.FloatTensor([0, sorted_arr[0].item()]), torch.FloatTensor([len(sorted_arr), sorted_arr[-1].item()])]
    _points = torch.cat([torch.arange(len(sorted_arr)).to(torch.float)[:, None], sorted_arr[:, None]], dim=1)
    ods = compute_orthogonal_distances(_points, _line)
    max_od_idx = ods.argmax() + 1

    max_od_val = sorted_arr[max_od_idx] / factor

    return max_od_val

def compute_self_sim_min(self_sim: torch.Tensor, mode="min") -> torch.Tensor:
    mins = torch.empty(len(self_sim))

    for i in range(len(self_sim)):
        mins[i] = compute_min(self_sim[i])

    if mode == "avg":
        minimum = mins.mean()
    elif mode == "min":
        minimum = mins.min()
    else:
        raise NotImplementedError()

    return minimum


def compute_min(scores: torch.Tensor) -> torch.Tensor:
    return _compute_min(scores)


def _compute_min(scores: torch.Tensor) -> torch.Tensor:
    return scores.min()


def compute_sym_th(self_sim: torch.Tensor, mode="min", within=False) -> float:

    _factor = 1
    if within:
        _factor = 2

    minimum = compute_self_sim_min(self_sim, mode=mode)
    th = torch.cos(torch.acos(minimum) / (2*_factor)).item()

    return th


def _compute_th(self_sim: torch.Tensor, th_type="symmin", within=False) -> float:

    _factor = 1
    if within:
        _factor = 2

    if th_type == "symmin":
        th = compute_sym_th(self_sim, mode="min", within=within)
    elif th_type == "symavg":
        th = compute_sym_th(self_sim, mode="avg", within=within)
    # elif th_type == "random-approx":
    #     th = np.cos(
    #         np.arccos(1 / np.sqrt(1024)) / 2
    #     )  # FIXME: 1024 is hard-coded at the momoent; fix it
    elif th_type == "indep":
        th = np.cos(np.pi / (4*_factor))
    else:
        raise NotImplementedError()

    return th


def compute_th(
    self_sim: torch.Tensor,
    num_bootstrapping=1,
    subsampling_ratio=1.0,
    th_type="sym-min",
    within=False,
) -> float:

    if num_bootstrapping == 1:
        th = _compute_th(self_sim, th_type=th_type, within=within)
        return th

    assert num_bootstrapping > 1
    n = len(self_sim)
    b = round(subsampling_ratio * n)

    ths = []
    for _ in range(num_bootstrapping):
        _idxes = torch.randint(0, n, (b,))
        _self_sim = self_sim[_idxes, :]
        _self_sim = _self_sim[:, _idxes]
        _th = _compute_th(_self_sim, th_type=th_type, within=within)
        ths.append(_th)

    th = np.mean(ths)

    return th


def sample_few_shot(
    X: torch.Tensor,
    fea_map_shape: bool = None,
    th_type: str = "indep",
    vote_type: str = "mean",
    num_bootstrapping: int = 1,
    subsampling_ratio: float = 1.0,
    return_class_sizes: bool = False,
):
    if fea_map_shape is not None:
        return _sample_patchwise_few_shot(
            X,
            fea_map_shape=fea_map_shape,
            th_type=th_type,
            vote_type=vote_type,
            num_bootstrapping=num_bootstrapping,
            subsampling_ratio=subsampling_ratio,
            return_class_sizes=return_class_sizes,
        )

    return _sample_few_shot(
        X,
        th_type=th_type,
        vote_type=vote_type,
        num_bootstrapping=num_bootstrapping,
        subsampling_ratio=subsampling_ratio,
        return_class_sizes=return_class_sizes,
    )


def _sample_few_shot(
    X: torch.Tensor,
    th_type="sym-min",
    vote_type="mean",
    num_bootstrapping=1,
    subsampling_ratio=1.0,
    return_class_sizes=False,
) -> torch.Tensor:
    self_sim = compute_self_sim(X, normalize=True)
    th = compute_th(
        self_sim,
        num_bootstrapping=num_bootstrapping,
        subsampling_ratio=subsampling_ratio,
        th_type=th_type,
    )
    class_sizes = predict_class_sizes(self_sim, th=th, vote_type=vote_type)
    if return_class_sizes:
        return class_sizes

    is_few_shot = predict_few_shot_class_samples(class_sizes)

    sample_indices = torch.where(is_few_shot == 1)[0]

    return X[sample_indices], sample_indices


def _sample_patchwise_few_shot(
    features: torch.Tensor,
    fea_map_shape,
    th_type="random-ideal-within",
    vote_type="mean",
    num_bootstrapping=1,
    subsampling_ratio=1.0,
    return_class_sizes=False,
):
    """
    Args:
        features: (n*h*w, c). To revert it back to feature map, the below is required.
        (n*h*w, c) -> (n, h*w, c) -> (n, c, h*w) -> (n, c, h, w)
    """
    fea_dim = features.shape[1]
    h, w = fea_map_shape[0], fea_map_shape[1]
    num_pixels = h * w

    patchwise_features = features.reshape(-1, num_pixels, fea_dim).permute(
        1, 0, 2
    )  # (num_patches, n, c)
    b = patchwise_features.shape[1]

    patchwise_self_sim = compute_self_sim(patchwise_features, normalize=True)

    patchwise_class_sizes = torch.empty((b, num_pixels), dtype=torch.float)
    patchwise_few_shot_idxes = torch.empty((b, num_pixels), dtype=torch.long)

    def _compute_patchwise_few_shot_idxes(p):
        _self_sim = patchwise_self_sim[p]
        _th = compute_th(
            _self_sim,
            num_bootstrapping=num_bootstrapping,
            subsampling_ratio=subsampling_ratio,
            th_type=th_type,
            within=True
        )

        _class_sizes = predict_class_sizes(_self_sim, th=_th, vote_type=vote_type)
        patchwise_class_sizes[:, p] = _class_sizes

        if not return_class_sizes:
            _few_shot_idxes = predict_few_shot_class_samples(_class_sizes)
            patchwise_few_shot_idxes[:, p] = _few_shot_idxes

    for p in tqdm(range(num_pixels), desc="Computing patchwise few-shot idxes"):
        _compute_patchwise_few_shot_idxes(p)

    # Parallel(n_jobs=4)(delayed(_compute_pixelwise_few_shot_idxes)(p) for p in range(num_pixels))
    if return_class_sizes:
        return patchwise_class_sizes.reshape((-1,))

    sample_indices = patchwise_few_shot_idxes.reshape((-1,))
    sample_features = features[sample_indices]

    return sample_features, sample_indices


def compute_orthogonal_distances(points: torch.Tensor, line_points: list) -> torch.Tensor:

    points = points.numpy()

    # Extract coordinates of the line
    (x1, y1), (x2, y2) = line_points[0].numpy(), line_points[1].numpy()

    # Calculate the slope of the line (y = mx + c)
    dx, dy = x2 - x1, y2 - y1
    if dx == 0:  # Special case for vertical line
        return [abs(x - x1) for x, y in points]

    m = dy / dx
    c = y1 - m * x1
    m_perp = -1 / m  # Slope of the perpendicular line

    distances = []
    for x, y in tqdm(points):
        c_perp = y - m_perp * x

        # Find intersection
        x_intersect = (c_perp - c) / (m - m_perp)
        y_intersect = m * x_intersect + c

        # Distance from point to line
        distance = np.sqrt((x_intersect - x)**2 + (y_intersect - y)**2)
        distances.append(distance)

    distances = torch.FloatTensor(distances)

    return distances



if __name__ == "__main__":
    # Test the function
    # num_samples, h, w, c = 1624, 28, 28, 1024  # Example dimensions
    # fea_maps = np.random.rand(num_samples, h, w, c)
    # self_sim = compute_pixelwise_self_sim(fea_maps)

    # print("Shape of fea_maps:", fea_maps.shape)
    # print("Shape of self_sim:", self_sim.shape)
    print("above requires revision")
