import torch
import random
import abc
import gc
import time
import numpy as np
from copy import deepcopy
from typing import Union
from tqdm import tqdm
from copy import deepcopy
from sklearn.neighbors import LocalOutlierFactor
from . import class_size, utils


class BaseSampler(abc.ABC):

    def __init__(self, percentage: float):
        if not 0 <= percentage <= 1:
            raise ValueError("Percentage value not in [0, 1].")
        self.percentage = percentage

    @abc.abstractmethod
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        pass

    def _store_type(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        self.features_is_numpy = isinstance(features, np.ndarray)
        if not self.features_is_numpy:
            self.features_device = features.device

    def _restore_type(self, features: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        if self.features_is_numpy:
            return features.cpu().numpy()
        return features.clone().to(self.features_device)


class IncrementalGreedyCoresetSampler(BaseSampler):

    def __init__(
        self,
        percentage: float,
        dimension_to_project_features_to: int,
        device: torch.device,
        brute: bool = True,
        max_coreset_shape=None,
    ):
        """Greedy Coreset sampling base class."""
        super().__init__(percentage)

        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to
        self.brute = brute

        if max_coreset_shape is not None:
            self.max_coreset_size = max_coreset_shape[0]
            self._max_coreset = torch.zeros(
                (int(max_coreset_shape[0] * (1 + percentage)), max_coreset_shape[1])
            )

    # FIXME: refactor
    def run(
        self,
        features: torch.Tensor,
        base_coreset: torch.Tensor = None,
    ) -> torch.Tensor:
        """Subsamples features using Greedy Coreset.

        Args:
            features: [N x D]
        """
        if self.percentage == 1:
            return features

        assert isinstance(features, torch.Tensor)
        assert isinstance(base_coreset, (torch.Tensor, type(None)))

        if base_coreset is not None:
            tqdm.write(f"coreset size: {len(base_coreset)}")

        if hasattr(self, "max_coreset_size"):
            base_coreset = self._suppress_coreset(base_coreset, num_repeats=100)

        reduced_features, reduced_base_coreset = self._reduce_features(
            features, base_features=base_coreset
        )
        sample_indices = self._compute_greedy_coreset_indices(
            reduced_features, base_coreset=reduced_base_coreset
        )
        sample_features = features[sample_indices]

        if hasattr(self, "_max_coreset"):
            n_features = len(sample_features)
            n_base_coreset = len(base_coreset) if base_coreset is not None else 0

            self._max_coreset[n_base_coreset : n_base_coreset + n_features] = (
                sample_features
            )
            sample_features = self._max_coreset[: n_features + n_base_coreset]
        else:
            if base_coreset is not None:
                sample_features = torch.cat([sample_features, base_coreset], dim=0)

        return sample_features, sample_indices

    def _reduce_features(
        self, features: torch.Tensor, base_features: torch.Tensor = None
    ):
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        )
        _ = mapper.to(self.device)
        features = features.to(self.device)
        with torch.no_grad():
            reduced_features = mapper(features).cpu()

        if base_features is not None:
            base_features = base_features.to(self.device)
            with torch.no_grad():
                reduced_base_features = mapper(base_features).cpu()
        else:
            reduced_base_features = None

        return reduced_features, reduced_base_features

    def _compute_greedy_coreset_indices(
        self, features: torch.Tensor, base_coreset=None
    ) -> torch.Tensor:
        if self.brute:
            return self._brute_compute_greedy_coreset_indices(features, base_coreset)
        return self._parallel_compute_greedy_coreset_indices(features, base_coreset)

    def _brute_compute_greedy_coreset_indices(
        self, features: torch.Tensor, base_coreset=None
    ) -> torch.Tensor:
        """Runs iterative greedy coreset selection.

        Args:
            features: [NxD] input feature bank to sample.
        """
        # torch.cuda.empty_cache()
        # gc.collect()
        print("WARNING: Brute-force greedy-sampling.")
        features = features.to(self.device)
        num_coreset_samples = int(len(features) * self.percentage)
        coreset_indices = [np.random.randint(len(features))]

        if base_coreset is not None:
            base_coreset = base_coreset.to(self.device)
            min_distances = torch.min(
                torch.cdist(features[None, :, :], base_coreset[None, :, :])[0], dim=1
            )[0]
        else:
            min_distances = torch.ones(len(features)).to(features.device) * float("inf")

        for _ in tqdm(
            range(1, num_coreset_samples), desc="brute force greedy sample", leave=False
        ):
            current_distances = torch.norm(
                features - features[coreset_indices[-1]], dim=1
            )
            min_distances = torch.minimum(min_distances, current_distances)
            coreset_indices.append(torch.argmax(min_distances, dim=0).item())
            current_distances = current_distances.cpu()
            del current_distances
            gc.collect()
            torch.cuda.empty_cache()

        return coreset_indices

    def _parallel_compute_greedy_coreset_indices(
        self, features: torch.Tensor, base_coreset: torch.Tensor = None
    ) -> torch.Tensor:
        """Runs iterative greedy coreset selection.

        Args:
            features: [NxD] input feature bank to sample.
        """
        distance_matrix = self._compute_batchwise_differences(features, features)
        coreset_anchor_distances = torch.norm(
            distance_matrix, dim=1
        )  # TODO: reason why? dummy to initialize?

        if base_coreset is not None:
            coreset_anchor_distances = self._compute_set_distances(
                features, base_coreset, num_repeats=1
            )

        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        for _ in range(num_coreset_samples):
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)

            coreset_select_distance = distance_matrix[
                :, select_idx : select_idx + 1  # noqa E203
            ]
            coreset_anchor_distances = torch.cat(
                [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
            )
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values

        return coreset_indices

    def _compute_batchwise_differences(
        self, matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """Computes batchwise Euclidean distances using PyTorch."""

        matrix_a = matrix_a.to(self.device)
        matrix_b = matrix_b.to(self.device)

        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt().cpu()

    def _suppress_coreset(
        self, coreset: torch.Tensor, reduction_rate=0.25, num_repeats: int = None
    ) -> torch.Tensor:
        assert isinstance(self.max_coreset_size, int)

        if coreset is None:
            return coreset

        coreset_size = len(coreset)
        assert isinstance(coreset, torch.Tensor)

        if len(coreset) <= self.max_coreset_size:
            return coreset

        required_size = max(int(self.max_coreset_size * (1 - reduction_rate)), 1)

        print(f"max_coreset_size: {self.max_coreset_size}")
        print(f"coreset size BEFORE suppression: {coreset_size}")

        intra_set_dists = self._compute_intra_set_distances(
            coreset, num_repeats=num_repeats
        )
        n_feas_to_remove = coreset_size - required_size
        idxes_to_remove = torch.topk(-intra_set_dists, n_feas_to_remove).indices
        mask = torch.ones(coreset_size, dtype=torch.bool)
        mask[idxes_to_remove] = False

        coreset = coreset[mask]

        print(f"coreset size AFTER suppression: {len(coreset)}")

        return coreset

    def _compute_intra_set_distances(
        self, features: torch.Tensor, num_repeats: int = None
    ) -> torch.Tensor:
        n = len(features)
        batch_size = features.shape[-1]

        batch_indices = [(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]
        indices = list(range(n))
        random.shuffle(indices)
        batch_indices = [
            indices[i : min(i + batch_size, n)] for i in range(0, n, batch_size)
        ]

        intra_set_distances = []

        # Loop over each batch with tqdm
        for batch_idx in tqdm(batch_indices, desc="Processing batches", leave=False):
            batch_features = features[batch_idx]
            distances = self._compute_set_distances(
                batch_features, features, k=2, num_repeats=num_repeats
            )
            intra_set_distances.append(distances)

        intra_set_distances = torch.cat(intra_set_distances)

        return intra_set_distances

    def _compute_set_distances(
        self,
        features: torch.Tensor,
        coreset: torch.Tensor,
        num_repeats: int = None,
        k: int = 1,
    ) -> torch.Tensor:
        num_coreset_features = len(coreset)

        def divide_to_parts(input_list, target_len):
            return [
                input_list[i : i + target_len]
                for i in range(0, len(input_list), target_len)
            ]

        list_idxes = divide_to_parts(
            torch.randperm(num_coreset_features), len(features)
        )

        if num_repeats is None:
            num_repeats = len(list_idxes)

        for i in range(num_repeats):
            distance_matrix = self._compute_batchwise_differences(
                features, coreset[list_idxes[i]]
            )
            if k > 1:
                distances = -torch.topk(-distance_matrix, k, dim=1).values[:, -1]
            else:
                distances = torch.min(distance_matrix, dim=1).values
            if i == 0:
                min_distances = distances
            else:
                min_distances = torch.minimum(min_distances, distances)

        return min_distances


class GreedyCoresetSampler(BaseSampler):

    def __init__(
        self,
        percentage: float,
        dimension_to_project_features_to: int,
        device: torch.device,
        brute: bool = True,
    ):
        """Greedy Coreset sampling base class."""
        super().__init__(percentage)

        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to
        self.brute = brute

        assert self.brute

    # FIXME: refactor
    def run(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Subsamples features using Greedy Coreset.

        Args:
            features: [N x D]
        """
        if self.percentage == 1:
            return features

        assert isinstance(features, torch.Tensor)

        reduced_features = self._reduce_features(features)
        sample_indices = self._compute_greedy_coreset_indices(reduced_features)
        sample_features = features[sample_indices]

        return sample_features, sample_indices

    def _reduce_features(self, features: torch.Tensor):
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        )
        _ = mapper.to(self.device)
        features = features.to(self.device)
        with torch.no_grad():
            reduced_features = mapper(features).cpu()

        del mapper

        return reduced_features

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> torch.Tensor:
        if self.brute:
            return self._brute_compute_greedy_coreset_indices(features)

        return self._parallel_compute_greedy_coreset_indices(features)

    def _brute_compute_greedy_coreset_indices(
        self, features: torch.Tensor
    ) -> torch.Tensor:
        """Runs iterative greedy coreset selection.

        Args:
            features: [NxD] input feature bank to sample.
        """
        # torch.cuda.empty_cache()
        # gc.collect()
        print("WARNING: Brute-force greedy-sampling.")
        features = features.to(self.device)
        num_coreset_samples = int(len(features) * self.percentage)
        coreset_indices = [np.random.randint(len(features))]

        min_distances = torch.ones(len(features)).to(features.device) * float("inf")

        for _ in tqdm(
            range(1, num_coreset_samples), desc="brute force greedy sample", leave=False
        ):
            current_distances = torch.norm(
                features - features[coreset_indices[-1]], dim=1
            )
            min_distances = torch.minimum(min_distances, current_distances)
            coreset_indices.append(torch.argmax(min_distances, dim=0).item())
            current_distances = current_distances.cpu()
            del current_distances
            gc.collect()
            torch.cuda.empty_cache()

        return coreset_indices

    def _parallel_compute_greedy_coreset_indices(
        self, features: torch.Tensor
    ) -> torch.Tensor:
        """Runs iterative greedy coreset selection.

        Args:
            features: [NxD] input feature bank to sample.
        """
        distance_matrix = self._compute_batchwise_differences(features, features)
        coreset_anchor_distances = torch.norm(
            distance_matrix, dim=1
        )  # TODO: reason why? dummy to initialize?

        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        for _ in range(num_coreset_samples):
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)

            coreset_select_distance = distance_matrix[
                :, select_idx : select_idx + 1  # noqa E203
            ]
            coreset_anchor_distances = torch.cat(
                [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
            )
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values

        return coreset_indices

    def _compute_batchwise_differences(
        self, matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """Computes batchwise Euclidean distances using PyTorch."""

        matrix_a = matrix_a.to(self.device)
        matrix_b = matrix_b.to(self.device)

        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt().cpu()


class LOFSampler(BaseSampler):
    def __init__(
        self,
        percentage: float = 0.85,  # 1.0
        lof_k: int = 6,  # 6
        dimension_to_project_features_to=128,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(percentage)

        self.lof_k = lof_k
        self.dimension_to_project_features_to = dimension_to_project_features_to
        self.device = device

    def run(
        self,
        features: torch.Tensor,
        feature_map_shape: torch.Tensor = None,
    ):
        # if feature_map_shape is provided, patchwise
        reduced_features = self._reduce_features(features)
        lof_scores = self._compute_lof_scores(
            reduced_features,
            feature_map_shape,
        )

        thresh = torch.quantile(lof_scores, self.percentage)
        sample_indices = torch.where(lof_scores < thresh)[0]

        sample_features = features[sample_indices]

        return sample_features, sample_indices

    def _compute_lof_scores(
        self,
        features,
        feature_map_shape=None,
    ):
        if feature_map_shape is not None:
            lof_scores = self._compute_patchwise_lof(features, feature_map_shape)
        else:
            lof_scores = self._compute_lof(features)

        return lof_scores

    def _compute_patchwise_lof(
        self, features: torch.Tensor, feature_map_shape
    ) -> torch.Tensor:
        h, w = feature_map_shape[0], feature_map_shape[1]
        features = features.reshape(-1, h * w, self.dimension_to_project_features_to)

        batch_size, num_patches, _ = (
            features.shape
        )  # batch_size, num_patches, embedding_dim

        clf = LocalOutlierFactor(n_neighbors=self.lof_k, metric="l2")

        scores = torch.empty((batch_size, num_patches), dtype=torch.float)

        for p in tqdm(range(num_patches), desc="Computing LOF..."):

            _features = features[:, p, :].cpu().numpy()
            clf.fit(_features)
            _scores = torch.FloatTensor(-clf.negative_outlier_factor_)
            scores[:, p] = _scores

        return scores.reshape((-1))

    def _compute_lof(self, features: torch.Tensor) -> torch.Tensor:

        clf = LocalOutlierFactor(n_neighbors=self.lof_k, metric="l2")
        clf.fit(features)
        scores = torch.FloatTensor(-clf.negative_outlier_factor_)

        return scores

    def _reduce_features(
        self,
        features: torch.Tensor,
    ):
        if features.shape[1] == self.dimension_to_project_features_to:
            return features

        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        )
        _ = mapper.to(self.device)
        features = features.to(self.device)
        with torch.no_grad():
            reduced_features = mapper(features).cpu()

        del mapper

        return reduced_features.clone().detach()


# class TailSampler(BaseSampler):
#     def __init__(
#         self,
#         th_type: "str" = "indep",
#         vote_type: "str" = "mean",
#     ):
#         self.th_type = th_type
#         self.vote_type = vote_type

#     def run(self, features: torch.Tensor, feature_map_shape: torch.Tensor = None):

#         if feature_map_shape is not None:
#             return self._run_patch(features, feature_map_shape)

#         return self._run(features)

#     def _compute_th(self, self_sim, within=False):
#         _th = class_size.compute_th(
#             self_sim,
#             th_type=self.th_type,
#             within=within
#         )
#         if not within:
#             self._th = _th
#         return _th

#     def _run(self, features: torch.Tensor):
#         self_sim = class_size.compute_self_sim(features, normalize=True)

#         th = self._compute_th(self_sim)

#         class_sizes = class_size.predict_class_sizes(self_sim, th=th, vote_type=self.vote_type)

#         is_few_shot = class_size.predict_few_shot_class_samples(class_sizes)

#         sample_indices = torch.where(is_few_shot == 1)[0]

#         return features[sample_indices], sample_indices

#     def _run_patch(self, features: torch.Tensor, feature_map_shape: torch.Tensor):
#         fea_dim = features.shape[1]
#         h, w = feature_map_shape[0], feature_map_shape[1]
#         num_pixels = h * w

#         patchwise_features = features.reshape(-1, num_pixels, fea_dim).permute(
#             1, 0, 2
#         )  # (num_patches, n, c)
#         b = patchwise_features.shape[1]

#         patchwise_self_sim = class_size.compute_self_sim(patchwise_features, normalize=True)

#         patchwise_class_sizes = torch.empty((b, num_pixels), dtype=torch.float)
#         patchwise_few_shot_idxes = torch.empty((b, num_pixels), dtype=torch.long)

#         def _compute_patchwise_few_shot_idxes(p):
#             _self_sim = patchwise_self_sim[p]
#             _th = self._compute_th(_self_sim, within=True)

#             _class_sizes = class_size.predict_class_sizes(_self_sim, th=_th, vote_type=self.vote_type)
#             patchwise_class_sizes[:, p] = _class_sizes

#         for p in tqdm(range(num_pixels), desc="Computing patchwise few-shot idxes"):
#             _compute_patchwise_few_shot_idxes(p)

#         sample_indices = patchwise_few_shot_idxes.reshape((-1,))
#         sample_features = features[sample_indices]

#         return sample_features, sample_indices

#     def return_th(self):
#         assert hasattr(self, '_th'), 'First, run.'
#         return self._th

# class CustomTailSampler(TailSampler):
#     def __init__(
#         self,
#         th,
#         vote_type: "str" = "mean",
#     ):
#         self.th = th
#         self.vote_type = vote_type

#     def run(self, features: torch.Tensor, feature_map_shape: torch.Tensor = None):

#         if feature_map_shape is not None:
#             return self._run_patch(features, feature_map_shape)

#         return self._run(features)

#     def _compute_th(self, self_sim, within=False):
#         _th = self._th
#         return _th


class TailSampler(BaseSampler):
    def __init__(
        self,
        th_type: "str" = "indep",
        vote_type: "str" = "mean",
    ):
        self.th_type = th_type
        self.vote_type = vote_type

    def run(self, features: torch.Tensor, feature_map_shape: torch.Tensor = None):

        return class_size.sample_few_shot(
            X=features,
            fea_map_shape=feature_map_shape,
            th_type=self.th_type,
            vote_type=self.vote_type,
        )


class FewShotLOFSampler(LOFSampler):

    def __init__(
        self,
        percentage: float = 0.85,  # 1.0
        lof_k: int = 6,  # 6
        dimension_to_project_features_to=128,
        device: torch.device = torch.device("cpu"),
        auto: bool = False,
    ):
        super().__init__(percentage, lof_k, dimension_to_project_features_to, device)

        self.auto = auto

    def run(
        self,
        features: torch.Tensor,
        feature_map_shape: torch.Tensor = None,
        th_type="indep",
    ):
        # if feature_map_shape is provided, patchwise
        reduced_features = self._reduce_features(features)

        class_sizes = class_size.sample_few_shot(
            reduced_features,
            feature_map_shape,
            th_type=th_type,
            return_class_sizes=True,
        )

        lof_scores = self._compute_lof_scores(
            reduced_features,
            feature_map_shape,
        )

        sample_indices = self._compute_sample_indices(lof_scores, class_sizes)

        sample_features = features[sample_indices]

        return sample_features, sample_indices

    def _compute_sample_indices(self, lof_scores, class_sizes):
        if self.auto:
            return self._compute_sample_indices_auto(lof_scores, class_sizes)

        return self._compute_sample_indices_manual(lof_scores, class_sizes)

    def _compute_sample_indices_auto_124(
        self, lof_scores: torch.Tensor, class_sizes: torch.Tensor
    ):

        lof_scores = lof_scores / lof_scores.max()
        few_shot_scores = 1 - class_sizes / class_sizes.max()
        coeffs = 1 - lof_scores * few_shot_scores

        modified_class_sizes = coeffs * class_sizes

        max_idx, K = class_size.detect_max_step_idx(class_sizes)

        percentage = max_idx / len(class_sizes)
        few_shot_lof_scores = lof_scores * few_shot_scores
        thresh = torch.quantile(few_shot_lof_scores, percentage)
        sample_indices = torch.where(few_shot_lof_scores < thresh)[0]

        return sample_indices

    # def _compute_sample_indices_auto_621_42487(self, lof_scores: torch.Tensor, class_sizes:torch.Tensor):
    def _compute_sample_indices_auto(
        self, lof_scores: torch.Tensor, class_sizes: torch.Tensor
    ):

        few_shot_scores = 1 - class_sizes / class_sizes.max()

        aug_lof_scores = lof_scores * few_shot_scores
        thresh = class_size.detect_max_step(
            aug_lof_scores, quantize=True, factor=len(aug_lof_scores)
        )
        sample_indices = torch.where(aug_lof_scores < thresh)[0]

        return sample_indices

    def _compute_sample_indices_auto_115_421986(
        self, lof_scores: torch.Tensor, class_sizes: torch.Tensor
    ):

        aug_class_sizes = class_sizes / lof_scores

        outlier_idxes = class_size.predict_few_shot_class_samples(aug_class_sizes)

        sample_indices = torch.where(outlier_idxes == 0)[0]

        return sample_indices

    def _compute_sample_indices_manual(
        self, lof_scores: torch.Tensor, class_sizes: torch.Tensor
    ):

        few_shot_scores = 1 - class_sizes / class_sizes.max()

        aug_lof_scores = lof_scores * few_shot_scores
        thresh = torch.quantile(aug_lof_scores, self.percentage)
        sample_indices = torch.where(aug_lof_scores < thresh)[0]

        return sample_indices


class SCSLOFSampler(LOFSampler):

    def __init__(
        self,
        th,
        percentage: float = 0.85,  # 1.0
        lof_k: int = 6,  # 6
        dimension_to_project_features_to=128,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(percentage, lof_k, dimension_to_project_features_to, device)

    def run(
        self,
        features: torch.Tensor,
        feature_map_shape: torch.Tensor = None,
    ):
        # if feature_map_shape is provided, patchwise
        reduced_features = self._reduce_features(features)

        class_sizes = class_size.sample_few_shot(
            reduced_features,
            feature_map_shape,
            th_type="indep",
            return_class_sizes=True,
        )

        lof_scores = self._compute_lof_scores(
            reduced_features,
            feature_map_shape,
        )

        sample_indices = self._compute_sample_indices(lof_scores, class_sizes)

        sample_features = features[sample_indices]

        return sample_features, sample_indices

    def _compute_sample_indices(self, lof_scores, class_sizes):
        return self._compute_sample_indices_auto(lof_scores, class_sizes)

    def _compute_sample_indices_manual(
        self, lof_scores: torch.Tensor, class_sizes: torch.Tensor
    ):

        few_shot_scores = 1 - class_sizes / class_sizes.max()

        aug_lof_scores = lof_scores * few_shot_scores
        thresh = torch.quantile(aug_lof_scores, self.percentage)
        sample_indices = torch.where(aug_lof_scores < thresh)[0]

        return sample_indices
