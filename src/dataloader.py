import numpy as np
import torch

import gc
import random

import src.utils as utils

from torch.utils.data import Subset, ConcatDataset, DataLoader

_DATASETS = {
    "mvtec": ["src.datasets.mvtec", "MVTecDataset"],
    "btad": ["src.datasets.btad", "BTADDataset"],
}


def clists():

    mvtec_clist = [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]

    visa_clist = [
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
    ]

    return mvtec_clist, visa_clist


def get_dataloaders(config, data_format, data_path, batch_size):

    if data_format == "mvtec-multiclass":
        return get_mvtec_dataloaders(
            data_path,
            batch_size,
            config.data.imagesize,
            config.data.resize,
            multiclass=True,
        )
    elif data_format == "labelme":
        return get_labelme_dataloaders(
            data_path,
            batch_size,
            config.data.imagesize,
            config.data.inputsize,
            config.data.overlap_ratio,
            config.data.roi,
        )
    else:
        raise NotImplementedError()


def get_mvtec_dataloaders(
    data_path, batch_size, imagesize, resize=None, multiclass=True
):

    import src.datasets.mvtec as mvtec

    classname_list = utils.get_folder_names(data_path)

    train_datasets = []
    test_dataloaders = []
    data_index = {}

    for classname in classname_list:
        _train_dataset = mvtec.MVTecDataset(
            source=data_path,
            classname=classname,
            resize=resize,
            imagesize=imagesize,
            split=mvtec.DatasetSplit.TRAIN,
        )

        _test_dataset = mvtec.MVTecDataset(
            source=data_path,
            classname=classname,
            resize=resize,
            imagesize=imagesize,
            split=mvtec.DatasetSplit.TEST,
        )

        data_index[classname] = len(_train_dataset)

        # Packaging
        train_datasets.append(_train_dataset)

        _test_dataloader = torch.utils.data.DataLoader(
            _test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )

        _test_dataloader.name = classname

        test_dataloaders.append(_test_dataloader)

    if multiclass:
        train_dataset = ConcatDataset(train_datasets)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )

        train_dataloader.name = "all"

        dataloaders = [
            {"train": train_dataloader, "test": test_dataloader}
            for test_dataloader in test_dataloaders
        ]

    else:
        dataloaders = []
        for _train_dataset, _test_dataloader in zip(train_datasets, test_dataloaders):
            _train_dataloader = torch.utils.data.DataLoader(
                _train_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
            )
            dataloaders.append({"train": _train_dataloader, "test": _test_dataloader})

    return dataloaders


def get_labelme_dataloaders(
    data_path, batch_size, imagesize, inputsize, overlap_ratio, roi
):
    from src.datasets.labelme import IterablePatchDataset

    train_data_path = f"{data_path}/train"
    test_data_path = f"{data_path}/val"

    train_dataset = IterablePatchDataset(
        train_data_path,
        patch_size=imagesize,
        input_size=inputsize,
        exclude_blob_area=True,
        overlap_ratio=overlap_ratio,
        roi=roi,
    )
    test_dataset = IterablePatchDataset(
        test_data_path,
        patch_size=imagesize,
        input_size=inputsize,
        exclude_blob_area=False,
        overlap_ratio=overlap_ratio,
        roi=roi,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=0, shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=0, shuffle=False
    )

    train_dataloader.name = "train"
    test_dataloader.name = "test"

    return [{"train": train_dataloader, "test": test_dataloader}]
