import os
import torch
import gc
import numpy as np
import GPUtil
from copy import deepcopy
from tqdm import tqdm

from .coreset_model import BaseCore
from .backbone import get_backbone
from .feature_embedder import FeatureEmbedder
from .coreset_model import get_coreset_model
from . import automl
from . import utils


class Engine:
    def __init__(
        self,
        config,
        backbone_name,
        device,
        input_shape,
        train_dataloader,
        test_dataloader,
        faiss_on_gpu,
        faiss_num_workers,
        sampler_on_gpu,
        save_dir_path,
        patch_infer,
        train_mode,
    ):
        self.config = config
        self.backbone_name = backbone_name
        self.device = device
        self.input_shape = input_shape
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.faiss_on_gpu = faiss_on_gpu
        self.faiss_num_workers = faiss_num_workers
        self.sampler_on_gpu = sampler_on_gpu
        self.save_dir_path = save_dir_path
        self.patch_infer = patch_infer
        self.train_mode = train_mode

    def train(self):
        if self.train_mode is None:
            self._basic_train()
        elif self.train_mode == "partition":
            self._partition_train()
        else:
            raise NotImplementedError()

    def _basic_train(self):
        backbone = get_backbone(self.backbone_name)

        # FIXME: move inside to coreset (it depends on coreset model)
        embedding_to_extract_from = None
        if hasattr(self.config.model, 'embedding_to_extract_from'):
            embedding_to_extract_from = self.config.model.embedding_to_extract_from 
        feature_embedder = FeatureEmbedder(
            device=self.device,
            input_shape=self.input_shape,
            backbone=backbone,
            layers_to_extract_from=self.config.model.layers_to_extract,
            embedding_to_extract_from=embedding_to_extract_from
        )

        coreset_model = get_coreset_model(
            self.config.model,
            feature_embedder=feature_embedder,
            imagesize=self.config.data.imagesize,
            device=self.device,
            faiss_on_gpu=self.faiss_on_gpu,
            faiss_num_workers=self.faiss_num_workers,
            sampler_on_gpu=self.sampler_on_gpu,
            save_dir_path=self.save_dir_path,
            brute=True,
        )

        coreset_model.fit(self.train_dataloader)

        self.coreset_model: BaseCore = coreset_model

    # FIXME: refactor
    def _partition_train(self):
        backbone = get_backbone(self.backbone_name)
        # FIXME: move inside to coreset (it depends on coreset model)
        feature_embedder = FeatureEmbedder(
            self.device, self.input_shape, backbone, self.config.model.layers_to_extract
        )

        partition_train_info = _get_partition_train_info(
            train_dataloader=self.train_dataloader,
            feature_map_shape=feature_embedder.get_feature_map_shape(),
            save_dir_path=self.save_dir_path,
            device=self.device,
        )

        len_feas = partition_train_info["len_feas"]
        data_partition_info = partition_train_info["data_partition_info"]

        fea_dim = feature_embedder.get_feature_map_shape()[-1]
        coreset_ratio = automl.get_max_coreset_ratio(fea_dim, len_feas)
        max_coreset_size = (
            automl.get_max_len_feas(fea_dim, max_usage=0.25, memory_type="available")
            if getattr(self.config.model, "suppress_coreset", False)
            else None
        )

        print(f"max_coreset_size: {max_coreset_size}")

        coreset_model = get_coreset_model(
            config=self.config,
            feature_embedder=feature_embedder,
            device=self.device,
            faiss_on_gpu=self.faiss_on_gpu,
            faiss_num_workers=self.faiss_num_workers,
            sampler_on_gpu=self.sampler_on_gpu,
            save_dir_path=self.save_dir_path,
            brute=False,
            coreset_ratio=coreset_ratio,
            max_coreset_size=max_coreset_size,
        )

        _fit_coreset_by_partition_train(
            coreset_model=coreset_model,
            train_dataloader=self.train_dataloader,
            data_partition_info=data_partition_info,
            save_dir_path=self.save_dir_path,
        )

        self.coreset_model: BaseCore = coreset_model

    def infer(self) -> np.ndarray:
        if self.patch_infer:
            return self._patch_infer()
        else:
            return self._image_infer()

    def _image_infer(self) -> np.ndarray:
        image_scores = []
        score_masks = []
        image_paths = []

        with tqdm(
            self.test_dataloader,
            desc="Inferring...",
        ) as data_iterator:
            for data in data_iterator:
                images = data["image"]

                _image_scores, _score_masks = self.coreset_model.predict(images)

                image_scores.extend(_image_scores)
                score_masks.extend(_score_masks)
                image_paths.extend(data["image_path"])

        return image_scores, score_masks, image_paths

    def _patch_infer(self) -> np.ndarray:
        image_scores = {}
        score_masks = {}

        image_sizes = self.test_dataloader.dataset.image_sizes
        for _image_path, _image_size in image_sizes.items():
            image_scores[_image_path] = 0
            score_masks[_image_path] = np.zeros((_image_size[0], _image_size[1]))

        with tqdm(
            self.test_dataloader,
            desc="Inferring...",
        ) as data_iterator:
            for _data in data_iterator:
                _images = _data["image"]
                _patches = _data["patch"]
                _image_paths = _data["image_path"]
                _image_scores, _score_masks = self.coreset_model.predict(_images)

                for i, _image_path in enumerate(_image_paths):
                    image_scores[_image_path] = max(
                        _image_scores[i], image_scores[_image_path]
                    )
                    _x0, _y0, _x1, _y1 = _patches[i]
                    score_masks[_image_path][_y0:_y1, _x0:_x1] = np.maximum(
                        score_masks[_image_path][_y0:_y1, _x0:_x1], _score_masks[i]
                    )  # FIXME: debug score_masks[_image_path][_y0:_y1, _x0:_x1] = _score_masks[i]

        return (
            list(image_scores.values()),
            list(score_masks.values()),
            list(image_scores.keys()),
        )


def _read_integer(filepath):
    with open(filepath, "r") as file:
        int_read = int(file.read())
    return int_read


def _save_integer(filepath, integer):
    with open(filepath, "w") as file:
        file.write(str(integer))


def _get_partition_train_info(
    train_dataloader, feature_map_shape, save_dir_path, **kwargs
):
    device = kwargs["device"]
    partition_train_info_path = os.path.join(save_dir_path, "partition_train_info.pkl")

    # FIXME: revise the interface to simplify and clean the below
    from .datasets.labelme import IterablePatchDataset

    assert isinstance(
        train_dataloader.dataset, IterablePatchDataset
    ), f"train_mode=partition works only for {IterablePatchDataset.__name__}"

    image_paths = deepcopy(train_dataloader.dataset.image_paths)

    if os.path.exists(partition_train_info_path):
        return utils.load_dict(partition_train_info_path)

    else:
        patch_partitions = {}

        available_gpu_memory = (
            GPUtil.getGPUs()[device.index].memoryFree * 1024**2 * 0.75
        )  # in byte
        # FIXME: refactor the below block
        len_feas = 0
        for image_path in image_paths:
            num_patches_per_image = len(train_dataloader.dataset.patch_dict[image_path])
            len_feas_per_image = (
                feature_map_shape[0] * feature_map_shape[1] * num_patches_per_image
            )
            num_distance_map = 4
            required_memory = len_feas_per_image**2 * 4 * num_distance_map

            n_divide = int(np.ceil(np.sqrt(required_memory / (available_gpu_memory))))

            patch_idxes = list(range(num_patches_per_image))
            np.random.shuffle(patch_idxes)
            patch_partitions[image_path] = np.array_split(patch_idxes, n_divide)
            print(
                f"avg. num. of patches in each partition: {np.array([len(partition) for partition in patch_partitions[image_path]]).mean()}"
            )

            len_feas += len_feas_per_image

        _train_dataloader = deepcopy(train_dataloader)

        data_partition_info = []
        for image_path in tqdm(image_paths, desc="getting data partition info..."):
            for patch_partition in patch_partitions[image_path]:
                assert len(patch_partition) > 1

                _data_info = {
                    "image_paths": [image_path],
                    "patch_dict": {
                        image_path: [
                            _train_dataloader.dataset.patch_dict[image_path][idx]
                            for idx in patch_partition
                        ]
                    },
                    "is_anomaly": {
                        image_path: _train_dataloader.dataset.is_anomaly[image_path]
                    },
                    "image_sizes": {
                        image_path: _train_dataloader.dataset.image_sizes[image_path]
                    },
                    "blob_areas_dict": {
                        image_path: _train_dataloader.dataset.blob_areas_dict[
                            image_path
                        ]
                    },
                }

                data_partition_info.append(_data_info)

        partition_train_info = {
            # "patch_partitions": patch_partitions,
            "len_feas": len_feas,
            "data_partition_info": data_partition_info,
        }

        utils.save_dict(partition_train_info, partition_train_info_path)

    return partition_train_info


def _fit_coreset_by_partition_train(
    coreset_model, train_dataloader, data_partition_info, save_dir_path
):
    _tmp_dataloader = deepcopy(train_dataloader)

    train_status_file_path = os.path.join(save_dir_path, "train_status.txt")
    if os.path.exists(train_status_file_path):
        idx_train_complete = _read_integer(train_status_file_path)
    else:
        idx_train_complete = 0

    for i in tqdm(
        range(idx_train_complete, len(data_partition_info)),
        desc="partition training...",
    ):
        _data_info = data_partition_info[i]

        _tmp_dataloader.dataset.image_paths = _data_info["image_paths"]
        _tmp_dataloader.dataset.patch_dict = _data_info["patch_dict"]
        _tmp_dataloader.dataset.is_anomaly = _data_info["is_anomaly"]
        _tmp_dataloader.dataset.image_sizes = _data_info["image_sizes"]
        _tmp_dataloader.dataset.blob_areas_dict = _data_info["blob_areas_dict"]

        if i == 0:
            coreset_model.fit(_tmp_dataloader, set_predictor=False)
        elif (i % 10 == 0) or (i == len(data_partition_info) - 1):
            coreset_model.incremental_fit(
                _tmp_dataloader, set_predictor=False, save_coreset=True
            )
            _save_integer(train_status_file_path, i + 1)
        else:
            coreset_model.incremental_fit(
                _tmp_dataloader, set_predictor=False, save_coreset=False
            )

    # just load
    if _read_integer(train_status_file_path) == len(data_partition_info):
        gc.collect()
        torch.cuda.empty_cache()
        coreset_model.fit(None)

    return coreset_model
