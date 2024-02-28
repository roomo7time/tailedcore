import abc

import gc
import torch
import numpy as np
import os
import shutil
import torch.nn.functional as F

from tqdm import tqdm
from copy import deepcopy
from torchvision import transforms

from torch.utils.data import DataLoader, SequentialSampler
import torchvision.transforms.functional as TF

from .feature_embedder import FeatureEmbedder
from .sampler import (
    IncrementalGreedyCoresetSampler,
    GreedyCoresetSampler,
    TailSampler,
    TailedLOFSampler,
    LOFSampler
)
from .common import FaissNN, NearestNeighbourScorer, RescaleSegmentor


class BaseCore(abc.ABC):

    @abc.abstractmethod
    def fit(self, trainloader):
        pass

    @abc.abstractmethod
    def predict(self, images):
        pass

    # @abc.abstractmethod
    # def incremental_fit(self, post_trainloader, set_predictor=True, save_coreset=True):
    #     pass


def get_coreset_model(
    model_config,
    feature_embedder,
    imagesize,
    device,
    faiss_on_gpu,
    faiss_num_workers,
    sampler_on_gpu,
    save_dir_path=None,
    brute=False,
    coreset_ratio=0.01,
    max_coreset_size=None,
    **kwargs,
) -> BaseCore:

    coreset_ratio = getattr(model_config, "greedy_ratio", coreset_ratio)
    if model_config.coreset_model_name == "patchcore":
        return ConstPatchCore(
            feature_embedder,
            device,
            imagesize,
            coreset_ratio=coreset_ratio,
            faiss_on_gpu=faiss_on_gpu,
            faiss_num_workers=faiss_num_workers,
            sampler_on_gpu=sampler_on_gpu,
            save_dir_path=save_dir_path,
            brute=brute,
            max_coreset_size=max_coreset_size,
        )
    elif model_config.coreset_model_name == "tailedpatch":
        return TailedPatch(
            feature_embedder,
            device,
            imagesize,
            coreset_ratio=coreset_ratio,
            faiss_on_gpu=faiss_on_gpu,
            faiss_num_workers=faiss_num_workers,
            sampler_on_gpu=sampler_on_gpu,
            save_dir_path=save_dir_path,
            brute=brute,
            noise_discriminate_on_tail_patches=getattr(model_config, 'noise_discriminate_on_tail_patches', False),
            noise_discriminator_type=getattr(model_config, 'noise_discriminator_type', 'few_shot_lof'),
            auto_thresholding_on_lof=getattr(model_config, 'auto_thresholding_on_lof', False)
        )
    elif model_config.coreset_model_name == "atailedpatch":
        return ATailedPatch(
            feature_embedder,
            device,
            imagesize,
            coreset_ratio=coreset_ratio,
            faiss_on_gpu=faiss_on_gpu,
            faiss_num_workers=faiss_num_workers,
            sampler_on_gpu=sampler_on_gpu,
            save_dir_path=save_dir_path,
            brute=brute,
            tail_th_type=model_config.tail_th_type,
            data_augment_tail=model_config.data_augment_tail,
            tail_lof=model_config.tail_lof,
        )
    elif model_config.coreset_model_name == "aatailedpatch":
        return AATailedPatch(
            feature_embedder,
            device,
            imagesize,
            coreset_ratio=coreset_ratio,
            faiss_on_gpu=faiss_on_gpu,
            faiss_num_workers=faiss_num_workers,
            sampler_on_gpu=sampler_on_gpu,
            save_dir_path=save_dir_path,
            brute=brute,
            tail_th_type=model_config.tail_th_type,
            tail_data_augment_type=model_config.tail_data_augment_type,
            tail_lof=model_config.tail_lof,
        )
    elif model_config.coreset_model_name == 'tailedsoftpatch':
        return TailedSoftPatch(
            feature_embedder,
            device,
            imagesize,
            coreset_ratio=coreset_ratio,
            faiss_on_gpu=faiss_on_gpu,
            faiss_num_workers=faiss_num_workers,
            sampler_on_gpu=sampler_on_gpu,
            save_dir_path=save_dir_path,
            brute=brute,
            tail_th_type=model_config.tail_th_type,
            tail_data_augment_type=model_config.tail_data_augment_type,
            tail_lof=model_config.tail_lof,
        )
    else:
        raise NotImplementedError()


class ConstPatchCore(BaseCore):

    def __init__(
        self,
        feature_embedder: FeatureEmbedder,
        device,
        imagesize,
        coreset_ratio=0.1,
        greedy_proj_dim=128,
        faiss_on_gpu=True,
        faiss_num_workers=8,
        sampler_on_gpu=True,
        anomaly_score_num_nn=1,
        save_dir_path=None,
        brute=True,
        max_coreset_size=None,
    ):
        super(ConstPatchCore, self).__init__()

        self.feature_embedder = feature_embedder

        if max_coreset_size is not None:
            max_coreset_shape = (
                max_coreset_size,
                feature_embedder.get_feature_map_shape()[-1],
            )
        else:
            max_coreset_shape = None

        self.sampler = IncrementalGreedyCoresetSampler(
            percentage=coreset_ratio,
            dimension_to_project_features_to=greedy_proj_dim,
            device=device if sampler_on_gpu else torch.device("cpu"),
            brute=brute,
            max_coreset_shape=max_coreset_shape,
        )

        nn_method = FaissNN(faiss_on_gpu, faiss_num_workers, device=device.index)
        self.anomaly_scorer = NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.feature_map_shape = self.feature_embedder.get_feature_map_shape()

        self.rescale_segmentor = RescaleSegmentor(device=device, target_size=imagesize)

        if save_dir_path:
            self.coreset_path = os.path.join(save_dir_path, "coreset.pt")
        else:
            self.coreset_path = None

    def fit(self, trainloader: DataLoader, set_predictor=True):

        tqdm.write("Fitting...")

        coreset_features = self._load_coreset_features()
        if coreset_features is None:
            coreset_features = self._get_coreset(trainloader)
            self._save_coreset_features(coreset_features)

        if set_predictor:
            self.anomaly_scorer.fit([coreset_features])

    def incremental_fit(self, trainloader, set_predictor=True, save_coreset=True):

        if hasattr(self, "_base_coreset"):
            base_coreset = self._base_coreset
        else:
            self._backup_coreset()
            base_coreset = self._load_coreset_features()
            self._base_coreset = base_coreset

        coreset_features = self._get_coreset(trainloader, base_coreset=base_coreset)

        self._base_coreset = coreset_features

        if save_coreset:
            self._save_coreset_features(coreset_features)

        if set_predictor:
            self.anomaly_scorer.fit([coreset_features])

    def _backup_coreset(self):
        backup_coreset_path = os.path.join(
            os.path.dirname(self.coreset_path), "coreset_backup.npy"
        )
        if os.path.exists(backup_coreset_path):
            return
        shutil.copyfile(self.coreset_path, backup_coreset_path)

    def _load_coreset_features(self) -> torch.Tensor:
        if self.coreset_path and os.path.exists(self.coreset_path):
            coreset_features = torch.load(self.coreset_path)
            tqdm.write("Loaded a saved coreset!")
            return coreset_features
        else:
            return None

    def _save_coreset_features(self, coreset_features: torch.Tensor):
        if self.coreset_path:
            tqdm.write(f"Saving a coreset at {self.coreset_path}")
            os.makedirs(os.path.dirname(self.coreset_path), exist_ok=True)
            torch.save(coreset_features.clone().cpu().detach(), self.coreset_path)
            tqdm.write("Saved a coreset!")

    def predict_on(self, testloader: DataLoader):
        features = self._get_features(testloader)
        image_scores, score_masks = self._get_scores(features)

        return image_scores, score_masks

    def _get_scores(self, features: np.ndarray) -> np.ndarray:
        batch_size = features.shape[0] // (
            self.feature_map_shape[0] * self.feature_map_shape[1]
        )
        _scores, _, _indices = self.anomaly_scorer.predict([features])

        scores = torch.from_numpy(_scores)

        image_scores = torch.max(scores.reshape(batch_size, -1), dim=-1).values
        patch_scores = scores.reshape(
            batch_size, self.feature_map_shape[0], self.feature_map_shape[1]
        )

        score_masks = self.rescale_segmentor.convert_to_segmentation(patch_scores)

        return image_scores.numpy().tolist(), score_masks

    def _get_features(self, dataloader: DataLoader) -> torch.Tensor:
        self.feature_embedder.eval()
        features = []
        with tqdm(
            dataloader, desc="Computing support features...", leave=False
        ) as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    images = data["image"]
                features.append(self.feature_embedder(images))

        features = torch.cat(features, dim=0)
        return features

    def _get_coreset(self, trainloader: DataLoader, base_coreset=None) -> torch.Tensor:
        features = self._get_features(trainloader)
        coreset_features, _ = self.sampler.run(features, base_coreset=base_coreset)

        return coreset_features

    def predict(self, images) -> np.ndarray:
        with torch.no_grad():
            features = self.feature_embedder(images)
        image_scores, score_masks = self._get_scores(np.array(features))

        return image_scores, score_masks


class TailedPatch(BaseCore):

    def __init__(
        self,
        feature_embedder: FeatureEmbedder,
        device,
        imagesize,
        coreset_ratio=0.01,
        greedy_proj_dim=128,
        faiss_on_gpu=True,
        faiss_num_workers=8,
        sampler_on_gpu=True,
        anomaly_score_num_nn=1,
        save_dir_path=None,
        brute=True,
        noise_discriminate_on_tail_patches=False,
        auto_thresholding_on_lof=False,
        noise_discriminator_type='few_shot_lof'
    ):
        super(TailedPatch, self).__init__()

        self.feature_embedder = feature_embedder

        self.greedy_coreset_sampler = GreedyCoresetSampler(
            percentage=coreset_ratio,
            dimension_to_project_features_to=greedy_proj_dim,
            device=device if sampler_on_gpu else torch.device("cpu"),
            brute=brute,
        )

        self.tail_sampler = TailSampler()

        if noise_discriminator_type == 'few_shot_lof':
            self.noise_discriminator = FewShotLOFSampler(
                device=device if sampler_on_gpu else torch.device("cpu"),
                auto=auto_thresholding_on_lof
            )
        elif  noise_discriminator_type == 'lof':
            self.noise_discriminator = LOFSampler(
                device=device if sampler_on_gpu else torch.device("cpu"),
            )

        self.faiss_on_gpu = faiss_on_gpu
        self.faiss_num_workers = faiss_num_workers
        self.device = device
        self.anomaly_score_num_nn = anomaly_score_num_nn

        self.feature_map_shape = self.feature_embedder.get_feature_map_shape()

        self.rescale_segmentor = RescaleSegmentor(device=device, target_size=imagesize)

        if save_dir_path:
            self.coreset_path = os.path.join(save_dir_path, "coreset.pt")
        else:
            self.coreset_path = None

        self.noise_discriminate_on_tail_patches = noise_discriminate_on_tail_patches

    def fit(self, trainloader: DataLoader, set_predictor=True):

        tqdm.write("Fitting...")

        coreset_features = self._load_coreset_features()
        if coreset_features is None:
            coreset_features = self._get_coreset(trainloader)
            self._save_coreset_features(coreset_features)

        if set_predictor:

            nn_method = FaissNN(
                self.faiss_on_gpu, self.faiss_num_workers, device=self.device.index
            )
            self.anomaly_scorer = NearestNeighbourScorer(
                n_nearest_neighbours=self.anomaly_score_num_nn, nn_method=nn_method
            )

            self.anomaly_scorer.fit([coreset_features])

    def _load_coreset_features(self) -> torch.Tensor:
        if self.coreset_path and os.path.exists(self.coreset_path):
            coreset_features = torch.load(self.coreset_path)
            tqdm.write("Loaded a saved coreset!")
            return coreset_features
        else:
            return None

    def _save_coreset_features(self, coreset_features: torch.Tensor):
        if self.coreset_path:
            tqdm.write(f"Saving a coreset at {self.coreset_path}")
            os.makedirs(os.path.dirname(self.coreset_path), exist_ok=True)
            torch.save(coreset_features.clone().cpu().detach(), self.coreset_path)
            tqdm.write("Saved a coreset!")

    def predict_on(self, testloader: DataLoader):
        features = self._get_features(testloader)
        image_scores, score_masks = self._get_scores(features)

        return image_scores, score_masks

    def _get_scores(self, features: np.ndarray) -> np.ndarray:
        batch_size = features.shape[0] // (
            self.feature_map_shape[0] * self.feature_map_shape[1]
        )
        _scores, _, _indices = self.anomaly_scorer.predict([features])

        scores = torch.from_numpy(_scores)

        image_scores = torch.max(scores.reshape(batch_size, -1), dim=-1).values
        patch_scores = scores.reshape(
            batch_size, self.feature_map_shape[0], self.feature_map_shape[1]
        )

        score_masks = self.rescale_segmentor.convert_to_segmentation(patch_scores)

        return image_scores.numpy().tolist(), score_masks

    def _get_features(
        self, dataloader: DataLoader, return_embeddings: bool = False
    ) -> torch.Tensor:

        self.feature_embedder.eval()
        features = []
        embeddings = []
        with tqdm(
            dataloader, desc="Computing support features...", leave=False
        ) as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    images = data["image"]
                if return_embeddings:
                    _features, _embeddings = self.feature_embedder(
                        images, return_embeddings=True
                    )
                    embeddings.append(_embeddings[:, :, 0, 0])
                else:
                    _features = self.feature_embedder(images)
                features.append(_features)

        features = torch.cat(features, dim=0)

        if return_embeddings:
            embeddings = torch.cat(embeddings, dim=0)
            return features, embeddings

        return features

    def _get_coreset(self, trainloader: DataLoader) -> torch.Tensor:
        
        features, embeddings = self._get_features(trainloader, return_embeddings=True)
        coreset_tail_features = self._get_coreset_tail(features, embeddings)
        coreset_head_features = self._get_coreset_head(features)

        coreset_features = torch.cat(
            [coreset_tail_features, coreset_head_features], dim=0
        )

        return coreset_features
    
    def _get_coreset_tail(self, features, embeddings):

        h, w = self.feature_map_shape[0], self.feature_map_shape[1]

        _, tail_embedding_indices = self.tail_sampler.run(embeddings)
        tail_features = features.reshape(-1, h * w, features.shape[-1])[
            tail_embedding_indices
        ].reshape(-1, features.shape[-1])

        if self.noise_discriminate_on_tail_patches:
            tail_features, _ = self.noise_discriminator.run(tail_features)

        coreset_tail_features, _ = self.greedy_coreset_sampler.run(tail_features)

        return coreset_tail_features
    
    def _get_coreset_head(self, features):
        head_features, _ = self.noise_discriminator.run(
            features, self.feature_map_shape[:2]
        )
        coreset_head_features, _ = self.greedy_coreset_sampler.run(head_features)

        return coreset_head_features

    def predict(self, images) -> np.ndarray:
        with torch.no_grad():
            features = self.feature_embedder(images)
        image_scores, score_masks = self._get_scores(np.array(features))

        return image_scores, score_masks


class ATailedPatch(BaseCore):

    def __init__(
        self,
        feature_embedder: FeatureEmbedder,
        device,
        imagesize,
        coreset_ratio=0.01,
        greedy_proj_dim=128,
        faiss_on_gpu=True,
        faiss_num_workers=8,
        sampler_on_gpu=True,
        anomaly_score_num_nn=1,
        save_dir_path=None,
        brute=True,
        tail_th_type: str = 'symmin',   # symmin, indep
        data_augment_tail: bool = False,
        tail_lof: bool = False,
    ):
        super(ATailedPatch, self).__init__()

        self.feature_embedder = feature_embedder

        self.greedy_coreset_sampler = GreedyCoresetSampler(
            percentage=coreset_ratio,
            dimension_to_project_features_to=greedy_proj_dim,
            device=device if sampler_on_gpu else torch.device("cpu"),
            brute=brute,
        )

        self.tail_sampler = TailSampler(th_type='symmin')
        if tail_lof:
            self.noise_discriminator = TailedLOFSampler(
                device=device if sampler_on_gpu else torch.device("cpu"),
                tail_th_type=tail_th_type
            )
        else:
            self.noise_discriminator = LOFSampler(
                device=device if sampler_on_gpu else torch.device("cpu"),
            )

        self.faiss_on_gpu = faiss_on_gpu
        self.faiss_num_workers = faiss_num_workers
        self.device = device
        self.anomaly_score_num_nn = anomaly_score_num_nn

        self.feature_map_shape = self.feature_embedder.get_feature_map_shape()

        self.rescale_segmentor = RescaleSegmentor(device=device, target_size=imagesize)

        if save_dir_path:
            self.coreset_path = os.path.join(save_dir_path, "coreset.pt")
        else:
            self.coreset_path = None
        
        self.tail_th_type = tail_th_type
        self.data_augment_tail = data_augment_tail


    def fit(self, trainloader: DataLoader, set_predictor=True):

        tqdm.write("Fitting...")

        coreset_features = self._load_coreset_features()
        if coreset_features is None:
            coreset_features = self._get_coreset(trainloader)
            self._save_coreset_features(coreset_features)

        if set_predictor:

            nn_method = FaissNN(
                self.faiss_on_gpu, self.faiss_num_workers, device=self.device.index
            )
            self.anomaly_scorer = NearestNeighbourScorer(
                n_nearest_neighbours=self.anomaly_score_num_nn, nn_method=nn_method
            )

            self.anomaly_scorer.fit([coreset_features])

    def _load_coreset_features(self) -> torch.Tensor:
        if self.coreset_path and os.path.exists(self.coreset_path):
            coreset_features = torch.load(self.coreset_path)
            tqdm.write("Loaded a saved coreset!")
            return coreset_features
        else:
            return None

    def _save_coreset_features(self, coreset_features: torch.Tensor):
        if self.coreset_path:
            tqdm.write(f"Saving a coreset at {self.coreset_path}")
            os.makedirs(os.path.dirname(self.coreset_path), exist_ok=True)
            torch.save(coreset_features.clone().cpu().detach(), self.coreset_path)
            tqdm.write("Saved a coreset!")

    def predict_on(self, testloader: DataLoader):
        features = self._get_features(testloader)
        image_scores, score_masks = self._get_scores(features)

        return image_scores, score_masks

    def _get_scores(self, features: np.ndarray) -> np.ndarray:
        batch_size = features.shape[0] // (
            self.feature_map_shape[0] * self.feature_map_shape[1]
        )
        _scores, _, _indices = self.anomaly_scorer.predict([features])

        scores = torch.from_numpy(_scores)

        image_scores = torch.max(scores.reshape(batch_size, -1), dim=-1).values
        patch_scores = scores.reshape(
            batch_size, self.feature_map_shape[0], self.feature_map_shape[1]
        )

        score_masks = self.rescale_segmentor.convert_to_segmentation(patch_scores)

        return image_scores.numpy().tolist(), score_masks

    def _get_features(
        self, dataloader: DataLoader, return_embeddings: bool = False
    ) -> torch.Tensor:

        self.feature_embedder.eval()
        features = []
        embeddings = []
        with tqdm(
            dataloader, desc="Computing support features...", leave=False
        ) as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    images = data["image"]
                if return_embeddings:
                    _features, _embeddings = self.feature_embedder(
                        images, return_embeddings=True
                    )
                    if _embeddings.ndim == 4:
                        _embeddings = _embeddings[:, :, 0, 0]
                    embeddings.append(_embeddings)
                else:
                    _features = self.feature_embedder(images)
                features.append(_features)

        features = torch.cat(features, dim=0)

        if return_embeddings:
            embeddings = torch.cat(embeddings, dim=0)
            return features, embeddings

        return features

    def _get_coreset(self, trainloader: DataLoader) -> torch.Tensor:
        
        features, embeddings = self._get_features(trainloader, return_embeddings=True)

        coreset_tail_features = self._get_coreset_tail(features, embeddings, trainloader)
        coreset_head_features = self._get_coreset_head(features)

        coreset_features = torch.cat(
            [coreset_tail_features, coreset_head_features], dim=0
        )

        return coreset_features

    def _get_coreset_tail(self, features, embeddings, trainloader):

        h, w = self.feature_map_shape[0], self.feature_map_shape[1]

        _, tail_embedding_indices = self.tail_sampler.run(embeddings)
        
        tail_base_features = features.reshape(-1, h * w, features.shape[-1])[
            tail_embedding_indices
        ].reshape(-1, features.shape[-1])

        tail_augmented_features = self._get_tail_augmented_features(trainloader, tail_embedding_indices)

        tail_features = torch.cat([tail_base_features, tail_augmented_features], dim=0)

        coreset_tail_features, _ = self.greedy_coreset_sampler.run(tail_features)

        return coreset_tail_features

    def _get_tail_augmented_features(self, trainloader: DataLoader, tail_indices):
        self.feature_embedder.eval()

        batch_size = trainloader.batch_size
        assert isinstance(trainloader.sampler, SequentialSampler)

        features = []
        tail_indices_set = set(tail_indices.tolist())  # Convert to a set for faster lookup

        with tqdm(trainloader, desc="Computing support features...", leave=False) as data_iterator:
            for batch_idx, data in enumerate(data_iterator):
                # Calculate the indices for the current batch
                current_indices = range(batch_idx * batch_size, (batch_idx + 1) * batch_size)

                # Check if these indices intersect with tail_indices
                relevant_indices = [idx for idx in current_indices if idx in tail_indices_set]

                if relevant_indices:
                    if isinstance(data, dict):
                        images = data["image"]

                    # Select only the images corresponding to relevant indices
                    relevant_images = images[[i - batch_idx * batch_size for i in relevant_indices]]

                    # Rotate each image by 45, 90, ..., 315 degrees and collect features
                    for angle in [0, 90, 180, 270]:
                        for flip in [0, 1]:
                            if (angle, flip) == (0, 0):
                                continue

                            if flip == 1:
                                flipped_images = TF.hflip(relevant_images)
                            else:
                                flipped_images = relevant_images
                            
                            if angle != 0:
                                rotated_images = TF.rotate(flipped_images, angle)
                            else:
                                rotated_images = flipped_images
                            
                            _features = self.feature_embedder(rotated_images)
                            features.append(_features)

        features = torch.cat(features, dim=0)

        return features
    
    def _get_coreset_head(self, features):
        head_features, _ = self.noise_discriminator.run(
            features, self.feature_map_shape[:2]
        )
        coreset_head_features, _ = self.greedy_coreset_sampler.run(head_features)

        return coreset_head_features

    def predict(self, images) -> np.ndarray:
        with torch.no_grad():
            features = self.feature_embedder(images)
        image_scores, score_masks = self._get_scores(np.array(features))

        return image_scores, score_masks
    

class AATailedPatch(BaseCore):

    def __init__(
        self,
        feature_embedder: FeatureEmbedder,
        device,
        imagesize,
        coreset_ratio=0.01,
        greedy_proj_dim=128,
        faiss_on_gpu=True,
        faiss_num_workers=8,
        sampler_on_gpu=True,
        anomaly_score_num_nn=1,
        save_dir_path=None,
        brute=True,
        tail_th_type: str = 'symmin',   # symmin, indep
        tail_data_augment_type: str = 'rot15flip',  # None, 'rot15flip', 'rot30flip'
        tail_lof: bool = False,
    ):
        super(AATailedPatch, self).__init__()

        self.feature_embedder = feature_embedder

        self.greedy_coreset_sampler = GreedyCoresetSampler(
            percentage=coreset_ratio,
            dimension_to_project_features_to=greedy_proj_dim,
            device=device if sampler_on_gpu else torch.device("cpu"),
            brute=brute,
        )

        self.tail_sampler = TailSampler(th_type='symmin')
        if tail_lof:
            self.noise_discriminator = TailedLOFSampler(
                device=device if sampler_on_gpu else torch.device("cpu"),
                tail_th_type=tail_th_type
            )
        else:
            self.noise_discriminator = LOFSampler(
                device=device if sampler_on_gpu else torch.device("cpu"),
            )

        self.faiss_on_gpu = faiss_on_gpu
        self.faiss_num_workers = faiss_num_workers
        self.device = device
        self.anomaly_score_num_nn = anomaly_score_num_nn

        self.feature_map_shape = self.feature_embedder.get_feature_map_shape()

        self.rescale_segmentor = RescaleSegmentor(device=device, target_size=imagesize)

        if save_dir_path:
            self.coreset_path = os.path.join(save_dir_path, "coreset.pt")
        else:
            self.coreset_path = None
        
        self.tail_th_type = tail_th_type
        self.tail_data_augment_type = tail_data_augment_type


    def fit(self, trainloader: DataLoader, set_predictor=True):

        tqdm.write("Fitting...")

        coreset_features = self._load_coreset_features()
        if coreset_features is None:
            coreset_features = self._get_coreset(trainloader)
            self._save_coreset_features(coreset_features)

        if set_predictor:

            nn_method = FaissNN(
                self.faiss_on_gpu, self.faiss_num_workers, device=self.device.index
            )
            self.anomaly_scorer = NearestNeighbourScorer(
                n_nearest_neighbours=self.anomaly_score_num_nn, nn_method=nn_method
            )

            self.anomaly_scorer.fit([coreset_features])

    def _load_coreset_features(self) -> torch.Tensor:
        if self.coreset_path and os.path.exists(self.coreset_path):
            coreset_features = torch.load(self.coreset_path)
            tqdm.write("Loaded a saved coreset!")
            return coreset_features
        else:
            return None

    def _save_coreset_features(self, coreset_features: torch.Tensor):
        if self.coreset_path:
            tqdm.write(f"Saving a coreset at {self.coreset_path}")
            os.makedirs(os.path.dirname(self.coreset_path), exist_ok=True)
            torch.save(coreset_features.clone().cpu().detach(), self.coreset_path)
            tqdm.write("Saved a coreset!")

    def predict_on(self, testloader: DataLoader):
        features = self._get_features(testloader)
        image_scores, score_masks = self._get_scores(features)

        return image_scores, score_masks

    def _get_scores(self, features: np.ndarray) -> np.ndarray:
        batch_size = features.shape[0] // (
            self.feature_map_shape[0] * self.feature_map_shape[1]
        )
        _scores, _, _indices = self.anomaly_scorer.predict([features])

        scores = torch.from_numpy(_scores)

        image_scores = torch.max(scores.reshape(batch_size, -1), dim=-1).values
        patch_scores = scores.reshape(
            batch_size, self.feature_map_shape[0], self.feature_map_shape[1]
        )

        score_masks = self.rescale_segmentor.convert_to_segmentation(patch_scores)

        return image_scores.numpy().tolist(), score_masks

    def _get_features(
        self, dataloader: DataLoader, return_embeddings: bool = False
    ) -> torch.Tensor:

        self.feature_embedder.eval()
        features = []
        embeddings = []
        with tqdm(
            dataloader, desc="Computing support features...", leave=False
        ) as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    images = data["image"]
                if return_embeddings:
                    _features, _embeddings = self.feature_embedder(
                        images, return_embeddings=True
                    )
                    if _embeddings.ndim == 4:
                        _embeddings = _embeddings[:, :, 0, 0]
                    embeddings.append(_embeddings)
                else:
                    _features = self.feature_embedder(images)
                features.append(_features)

        features = torch.cat(features, dim=0)

        if return_embeddings:
            embeddings = torch.cat(embeddings, dim=0)
            return features, embeddings

        return features

    def _get_coreset(self, trainloader: DataLoader) -> torch.Tensor:
        
        features, embeddings = self._get_features(trainloader, return_embeddings=True)

        coreset_tail_features = self._get_coreset_tail(features, embeddings, trainloader)
        coreset_head_features = self._get_coreset_head(features)

        coreset_features = torch.cat(
            [coreset_tail_features, coreset_head_features], dim=0
        )

        return coreset_features

    def _get_coreset_tail(self, features, embeddings, trainloader):

        h, w = self.feature_map_shape[0], self.feature_map_shape[1]

        _, tail_embedding_indices = self.tail_sampler.run(embeddings)

        tail_base_features = features.reshape(-1, h * w, features.shape[-1])[
            tail_embedding_indices
        ].reshape(-1, features.shape[-1])

        tail_augmented_features = self._get_tail_augmented_features(trainloader, tail_embedding_indices)

        tail_features = torch.cat([tail_base_features, tail_augmented_features], dim=0)

        coreset_tail_features, _ = self.greedy_coreset_sampler.run(tail_features)

        return coreset_tail_features

    def _get_tail_augmented_features(self, trainloader: DataLoader, tail_indices: torch.Tensor):

        if self.tail_data_augment_type is None:
            return None
        elif self.tail_data_augment_type == 'rot15flip':
            return self._get_tail_augmented_features_rotflip(trainloader, tail_indices, rot_degree=15, flip=True)
        elif self.tail_data_augment_type == 'rot30flip':
            return self._get_tail_augmented_features_rotflip(trainloader, tail_indices, rot_degree=30, flip=True)
        else:
            raise NotImplementedError()

    def _get_tail_augmented_features_rotflip(self, trainloader: DataLoader, tail_indices: torch.Tensor, rot_degree=15, flip=False):
        self.feature_embedder.eval()

        def _revise_trainloader(trainloader):   # hard-coded
            trainloader = deepcopy(trainloader)
            for i in range(len(trainloader.dataset.datasets)):
                center_crop_size = trainloader.dataset.datasets[i].transform_img.transforms[1].size
                trainloader.dataset.datasets[i].transform_img = transforms.Compose([
                    trainloader.dataset.datasets[0].transform_img.transforms[0],
                    trainloader.dataset.datasets[0].transform_img.transforms[-2],
                    trainloader.dataset.datasets[0].transform_img.transforms[-1]
                ])
            
            return trainloader, center_crop_size
        
        trainloader, center_crop_size = _revise_trainloader(trainloader)

        batch_size = trainloader.batch_size
        assert isinstance(trainloader.sampler, SequentialSampler)

        features = []
        tail_indices_set = set(tail_indices.tolist())  # Convert to a set for faster lookup

        if flip:
            flip_range = [0,1]
        else:
            flip_range = [0]

        with tqdm(trainloader, desc="Computing augmented features...", leave=False) as data_iterator:
            for batch_idx, data in enumerate(data_iterator):
                # Calculate the indices for the current batch
                current_indices = range(batch_idx * batch_size, (batch_idx + 1) * batch_size)

                # Check if these indices intersect with tail_indices
                relevant_indices = [idx for idx in current_indices if idx in tail_indices_set]
                
                if len(relevant_indices) == 0:
                    continue

                images = data["image"]

                # Select only the images corresponding to relevant indices
                relevant_images = images[[i - batch_idx * batch_size for i in relevant_indices]]

                for angle in range(0, 360, rot_degree):
                    for flip in flip_range:
                        if (angle, flip) == (0, 0):
                            continue

                        if flip == 1:
                            flipped_images = TF.hflip(relevant_images)
                        else:
                            flipped_images = relevant_images
                        
                        if angle != 0:
                            horizontal_padding, vertical_padding = calculate_padding_for_rotation(center_crop_size[1], center_crop_size[0])

                            padding = (horizontal_padding, horizontal_padding, vertical_padding, vertical_padding)

                            expanded_iamges = F.pad(flipped_images, padding, mode='replicate')
                            _rotated_images = TF.rotate(expanded_iamges, angle)
                        else:
                            _rotated_images = flipped_images
                        
                        rotated_images = TF.center_crop(_rotated_images, center_crop_size)
                        
                        _features = self.feature_embedder(rotated_images)
                        features.append(_features)

                        assert _features.shape[0] % (self.feature_map_shape[0]*self.feature_map_shape[1]) == 0

        features = torch.cat(features, dim=0)

        return features
    
    def _get_coreset_head(self, features):
        head_features, _ = self.noise_discriminator.run(
            features, self.feature_map_shape[:2]
        )
        coreset_head_features, _ = self.greedy_coreset_sampler.run(head_features)

        return coreset_head_features

    def predict(self, images) -> np.ndarray:
        with torch.no_grad():
            features = self.feature_embedder(images)
        image_scores, score_masks = self._get_scores(np.array(features))

        return image_scores, score_masks
    

class TailedSoftPatch(BaseCore):

    def __init__(
        self,
        feature_embedder: FeatureEmbedder,
        device,
        imagesize,
        coreset_ratio=0.01,
        greedy_proj_dim=128,
        faiss_on_gpu=True,
        faiss_num_workers=8,
        sampler_on_gpu=True,
        anomaly_score_num_nn=1,
        save_dir_path=None,
        brute=True,
        tail_th_type: str = 'symmin',   # symmin, indep
        tail_data_augment_type: str = None,
        tail_lof: bool = False,
    ):
        super(TailedSoftPatch, self).__init__()

        self.feature_embedder = feature_embedder

        self.greedy_coreset_sampler = GreedyCoresetSampler(
            percentage=coreset_ratio,
            dimension_to_project_features_to=greedy_proj_dim,
            device=device if sampler_on_gpu else torch.device("cpu"),
            brute=brute,
        )

        self.tail_sampler = TailSampler(th_type=tail_th_type)
        if tail_lof:
            self.noise_discriminator = TailedLOFSampler(
                device=device if sampler_on_gpu else torch.device("cpu"),
                tail_th_type=tail_th_type
            )
        else:
            self.noise_discriminator = LOFSampler(
                device=device if sampler_on_gpu else torch.device("cpu"),
            )

        self.faiss_on_gpu = faiss_on_gpu
        self.faiss_num_workers = faiss_num_workers
        self.device = device
        self.anomaly_score_num_nn = anomaly_score_num_nn

        self.feature_map_shape = self.feature_embedder.get_feature_map_shape()

        self.rescale_segmentor = RescaleSegmentor(device=device, target_size=imagesize)

        if save_dir_path:
            self.coreset_path = os.path.join(save_dir_path, "coreset.pt")
        else:
            self.coreset_path = None
        
        self.tail_th_type = tail_th_type
        self.tail_data_augment_type = tail_data_augment_type


    def fit(self, trainloader: DataLoader, set_predictor=True):

        tqdm.write("Fitting...")

        coreset_features = self._load_coreset_features()
        if coreset_features is None:
            coreset_features = self._get_coreset(trainloader)
            self._save_coreset_features(coreset_features)

        if set_predictor:

            nn_method = FaissNN(
                self.faiss_on_gpu, self.faiss_num_workers, device=self.device.index
            )
            self.anomaly_scorer = NearestNeighbourScorer(
                n_nearest_neighbours=self.anomaly_score_num_nn, nn_method=nn_method
            )

            self.anomaly_scorer.fit([coreset_features])

    def _load_coreset_features(self) -> torch.Tensor:
        if self.coreset_path and os.path.exists(self.coreset_path):
            coreset_features = torch.load(self.coreset_path)
            tqdm.write("Loaded a saved coreset!")
            return coreset_features
        else:
            return None

    def _save_coreset_features(self, coreset_features: torch.Tensor):
        if self.coreset_path:
            tqdm.write(f"Saving a coreset at {self.coreset_path}")
            os.makedirs(os.path.dirname(self.coreset_path), exist_ok=True)
            torch.save(coreset_features.clone().cpu().detach(), self.coreset_path)
            tqdm.write("Saved a coreset!")

    def predict_on(self, testloader: DataLoader):
        features = self._get_features(testloader)
        image_scores, score_masks = self._get_scores(features)

        return image_scores, score_masks

    def _get_scores(self, features: np.ndarray) -> np.ndarray:
        batch_size = features.shape[0] // (
            self.feature_map_shape[0] * self.feature_map_shape[1]
        )
        _scores, _, _indices = self.anomaly_scorer.predict([features])

        scores = torch.from_numpy(_scores)

        image_scores = torch.max(scores.reshape(batch_size, -1), dim=-1).values
        patch_scores = scores.reshape(
            batch_size, self.feature_map_shape[0], self.feature_map_shape[1]
        )

        score_masks = self.rescale_segmentor.convert_to_segmentation(patch_scores)

        return image_scores.numpy().tolist(), score_masks

    def _get_features(
        self, dataloader: DataLoader, return_embeddings: bool = False
    ) -> torch.Tensor:

        self.feature_embedder.eval()
        features = []
        embeddings = []
        with tqdm(
            dataloader, desc="Computing support features...", leave=False
        ) as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    images = data["image"]
                if return_embeddings:
                    _features, _embeddings = self.feature_embedder(
                        images, return_embeddings=True
                    )
                    if _embeddings.ndim == 4:
                        _embeddings = _embeddings[:, :, 0, 0]
                    embeddings.append(_embeddings)
                else:
                    _features = self.feature_embedder(images)
                features.append(_features)

        features = torch.cat(features, dim=0)

        if return_embeddings:
            embeddings = torch.cat(embeddings, dim=0)
            return features, embeddings

        return features

    def _get_coreset(self, trainloader: DataLoader) -> torch.Tensor:
        
        features, embeddings = self._get_features(trainloader, return_embeddings=True)

        tail_features = self._get_tail_features(features, embeddings, trainloader)
        repeat_factor = max(len(features) // len(tail_features), 1)

        if repeat_factor > 1:
            tail_features = tail_features.repeat(repeat_factor, 1)

        features = torch.cat(
            [tail_features, features], dim=0
        )

        features, _ = self.noise_discriminator.run(
            features, self.feature_map_shape[:2]
        )

        coreset_features, _ = self.greedy_coreset_sampler.run(features)

        return coreset_features
    
    def _get_tail_features(self, features, embeddings, trainloader):

        h, w = self.feature_map_shape[0], self.feature_map_shape[1]

        _, tail_embedding_indices = self.tail_sampler.run(embeddings)

        tail_base_features = features.reshape(-1, h * w, features.shape[-1])[
            tail_embedding_indices
        ].reshape(-1, features.shape[-1])

        tail_augmented_features = self._get_tail_augmented_features(trainloader, tail_embedding_indices)

        tail_features = torch.cat([tail_base_features, tail_augmented_features], dim=0)

        return tail_features
    
    def _get_tail_augmented_features(self, trainloader: DataLoader, tail_indices):
        if self.tail_data_augment_type == "rot15flip":
            return self._get_tail_augmented_features_rot15flip(trainloader, tail_indices)
        else:
            raise NotImplementedError()

    def _get_tail_augmented_features_rot15flip(self, trainloader: DataLoader, tail_indices: torch.Tensor):
        self.feature_embedder.eval()

        def _revise_trainloader(trainloader):   # hard-coded
            trainloader = deepcopy(trainloader)
            for i in range(len(trainloader.dataset.datasets)):
                center_crop_size = trainloader.dataset.datasets[i].transform_img.transforms[1].size
                trainloader.dataset.datasets[i].transform_img = transforms.Compose([
                    trainloader.dataset.datasets[0].transform_img.transforms[0],
                    trainloader.dataset.datasets[0].transform_img.transforms[-2],
                    trainloader.dataset.datasets[0].transform_img.transforms[-1]
                ])
            
            return trainloader, center_crop_size
        
        trainloader, center_crop_size = _revise_trainloader(trainloader)

        batch_size = trainloader.batch_size
        assert isinstance(trainloader.sampler, SequentialSampler)

        features = []
        tail_indices_set = set(tail_indices.tolist())  # Convert to a set for faster lookup

        with tqdm(trainloader, desc="Computing augmented features...", leave=False) as data_iterator:
            for batch_idx, data in enumerate(data_iterator):
                # Calculate the indices for the current batch
                current_indices = range(batch_idx * batch_size, (batch_idx + 1) * batch_size)

                # Check if these indices intersect with tail_indices
                relevant_indices = [idx for idx in current_indices if idx in tail_indices_set]
                
                if len(relevant_indices) == 0:
                    continue

                images = data["image"]

                # Select only the images corresponding to relevant indices
                relevant_images = images[[i - batch_idx * batch_size for i in relevant_indices]]

                for angle in range(0, 360, 15):
                    for flip in [0, 1]:
                        if (angle, flip) == (0, 0):
                            continue

                        if flip == 1:
                            flipped_images = TF.hflip(relevant_images)
                        else:
                            flipped_images = relevant_images
                        
                        if angle != 0:
                            horizontal_padding, vertical_padding = calculate_padding_for_rotation(center_crop_size[1], center_crop_size[0])

                            padding = (horizontal_padding, horizontal_padding, vertical_padding, vertical_padding)

                            expanded_iamges = F.pad(flipped_images, padding, mode='replicate')
                            _rotated_images = TF.rotate(expanded_iamges, angle)
                        else:
                            _rotated_images = flipped_images
                        
                        rotated_images = TF.center_crop(_rotated_images, center_crop_size)
                        
                        _features = self.feature_embedder(rotated_images)
                        features.append(_features)

                        assert _features.shape[0] % (self.feature_map_shape[0]*self.feature_map_shape[1]) == 0

        features = torch.cat(features, dim=0)

        return features
    
    def _get_head_features(self, features):
        head_features, _ = self.noise_discriminator.run(
            features, self.feature_map_shape[:2]
        )
        coreset_head_features, _ = self.greedy_coreset_sampler.run(head_features)

        return coreset_head_features

    def predict(self, images) -> np.ndarray:
        with torch.no_grad():
            features = self.feature_embedder(images)
        image_scores, score_masks = self._get_scores(np.array(features))

        return image_scores, score_masks



import math
def calculate_padding_for_rotation(width, height):
    # Calculate the diagonal length using the Pythagorean theorem
    diagonal = math.sqrt(width ** 2 + height ** 2)

    # Calculate horizontal and vertical padding
    horizontal_padding = (diagonal - width) / 2
    vertical_padding = (diagonal - height) / 2

    # Round to nearest integer since padding values must be integers
    return int(math.ceil(horizontal_padding)), int(math.ceil(vertical_padding))

import torchvision.transforms as transforms
def unnormalize_and_save(tensor, file_path='./tmp_img.jpg'):
    # Define the mean and std deviation for ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    # Unnormalize the image
    unnormalized_image = tensor.clone()  # Clone the tensor to avoid in-place modifications
    unnormalized_image = unnormalized_image * std + mean

    # Ensure the tensor is in the range [0, 1]
    unnormalized_image = torch.clamp(unnormalized_image, 0, 1)

    # Convert to PIL for saving
    pil_image = transforms.ToPILImage()(unnormalized_image)

    # Save the image
    pil_image.save(file_path, 'JPEG')