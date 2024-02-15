import torch
import random
import contextlib
import os
import random
import argparse
import tqdm

import numpy as np

import src.utils as utils
from src.dataloader import get_dataloader
from src.backbone import get_backbone
from src.feature_embedder import FeatureEmbedder
from src.coreset_model import get_coreset_model

DATA_PATH = "/home/jay/mnt/hdd01/data/mvtec_anomaly_detection_toy"


def parse_args():
    mvtec_clist, visa_clist = utils.clists()

    parser = argparse.ArgumentParser()

    ######################## CONFIGURE YOUR DATA ROOT! ########################
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    #####################################################################

    ####################### SAMPLER (MODEL) CONFIGURATIONS #######################
    parser.add_argument("--coreset_model_name", type=str, default="patchcore")
    # 2.1 Backbone
    parser.add_argument(
        "--backbone_names", type=str, action="append", default=["wideresnet50"]
    )
    parser.add_argument(
        "--layers_to_extract", type=str, action="append", default=["layer2", "layer3"]
    )
    # 2.2 Image and feature size
    parser.add_argument("--resize", default=256, type=int)
    parser.add_argument("--imagesize", default=224, type=int)
    # 2.2 Sampler
    parser.add_argument("--sampler_type", type=str, action="append", default=["greedy"])
    # 2.2.1 Greedy sampling params
    parser.add_argument("--greedy_ratio", type=float, action="append", default=[0.001])
    parser.add_argument("--greedy_proj_dim", type=int, action="append", default=[128])
    # 2.2.2 Local outlier factor params
    parser.add_argument(
        "--lof_thresh", type=int, action="append", default=[-1]
    )  # For lof types
    parser.add_argument(
        "--lof_k", type=int, action="append", default=[-1]
    )  # For lof types
    # 2.2.3 Ours
    # Score
    parser.add_argument("--normalize", type=str, action="append", default=[-1])
    parser.add_argument("--kthnnd_k", type=int, action="append", default=[-1])
    parser.add_argument("--kthnnd_p", type=float, action="append", default=[-1])
    parser.add_argument("--kthnnd_T", type=int, action="append", default=[-1])
    # Bound
    parser.add_argument("--cd_bound_p", type=float, action="append", default=[-1])
    parser.add_argument("--cd_bound_T", type=int, action="append", default=[-1])
    # Change Detect
    parser.add_argument("--cd_detect_p", type=float, action="append", default=[-1])
    parser.add_argument("--cd_detect_T", type=int, action="append", default=[-1])
    # 2.3 Seed
    parser.add_argument("--model_seed", type=int, default=0)
    #####################################################################

    ####################### EVAL CONFIGURATIONS #######################
    # 3.1 General
    parser.add_argument("--faiss_distance", type=str, default="L2")
    parser.add_argument("--faiss_k_neighbor", type=int, default=1)
    # 3.2 SoftPatch
    parser.add_argument("--weight", type=str, default=["False"])
    #####################################################################

    ####################### HARDWARE CONFIGURATIONS #######################
    parser.add_argument("--gpu", type=int, default=[0])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--faiss_on_gpu", type=bool, default=True)
    parser.add_argument("--faiss_num_workers", type=int, default=0)

    parser.add_argument("--sampler_on_gpu", type=bool, default=False)
    #####################################################################

    args = parser.parse_args()

    if "mvtec" in args.data_path:
        args.dataset = "mvtec"
        args.subdatasets = mvtec_clist
    if "visa" in args.data_path:
        args.dataset = "visa"
        args.subdatasets = visa_clist

    args.coreset_dir, args.result_dir = utils.dirs(args)

    return args


def main(args):
    if os.path.exists(f"{args.result_dir}/results.csv"):
        return

    set_seed(args.model_seed)

    device = utils.set_torch_device(args.gpu)
    # device_context = (
    #     torch.cuda.device("cuda:{}".format(device.index))
    #     if "cuda" in device.type.lower()
    #     else contextlib.suppress()
    # )

    input_shape = (3, args.imagesize, args.imagesize)

    # with device_context:
    # Load dataloader
    list_of_dataloaders, data_index = get_dataloader(args)

    for backbone_name in args.backbone_names:
        backbone = get_backbone(backbone_name)
        feature_embedder = FeatureEmbedder(
            device, input_shape, backbone, args.layers_to_extract
        )

    print()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    args = parse_args()
    main(args)
