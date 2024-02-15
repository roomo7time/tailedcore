import numpy as np
import random
import torch

import argparse
import contextlib
import os
import glob
import pickle

import PIL
from torchvision import transforms
from source.coreset import PatchMaker

import matplotlib.pyplot as plt

from source.coreset import get_cores, eval_sampling
from source.dataloader import get_dataloader
import source.utils as utils

from main_data import main_data_
from main_model import main_model_
from source.dataloader import get_dataloader


def parse_args():

    parser = argparse.ArgumentParser()

    # 1. Data
    # 1.0. Configure root for data
    parser.add_argument(
        "--DATA_ROOT", type=str, default="/home/jay/savespace/database/generic"
    )
    parser.add_argument(
        "--SYMLINK_ROOT",
        type=str,
        default="/home/jay/savespace/database/SYMLINKS/project_tailcaptor",
    )
    # 1.1. Choose dataset
    parser.add_argument("--dataset", type=str, default="mvtec")
    # 1.2. Configure Wildness (i.e., few shot and noise)
    ### Few shot
    parser.add_argument(
        "--few_shot_how", type=str, default="discrete"
    )  # "discrete" or "pareto"
    parser.add_argument(
        "--few_shot_class", type=float, default=0.6
    )  # when few_shot_how == "identical"
    parser.add_argument(
        "--few_shot_sample", type=int, default=4
    )  # when few_shot_how == "identical"
    parser.add_argument(
        "--few_shot_pareto_a", type=int, default=-1
    )  # when few_shot_how == "pareto"
    ### Noise
    parser.add_argument(
        "--noise_how", type=str, default="prior"
    )  # "prior" or "posterior"
    parser.add_argument("--noise_class", type=float, default=0.4)
    parser.add_argument("--noise_sample", type=float, default=0.1)
    ### Adding noise to few shot class
    parser.add_argument("--noise_addition_to_fewshot", type=int, default=0)
    ### Eval overlap on noise class
    parser.add_argument("--noise_eval_overlap", type=str, default="True")
    # 1.3. Size configuration
    parser.add_argument("--imagesize", default=224, type=int)
    parser.add_argument("--resize", default=256, type=int)

    # 2. Model
    # 2.1. Backbone
    parser.add_argument("--backbone_names", "-bn", type=str, default=["wideresnet50"])
    parser.add_argument(
        "--layers_to_extract", "-lte", type=str, default=["layer2", "layer3"]
    )
    # # 2.2. Greedy sampling
    # parser.add_argument("--greedy_ratio", type=float, default=1.0)
    # parser.add_argument("--greedy_dimension", type=int, default=128)
    # 2.2. Voter (GAPS)
    parser.add_argument("--few_shot_captor", type=str, default="GAP_tail_captor")
    parser.add_argument("--few_shot_greedy_ratio", type=float, default=0.1)
    parser.add_argument("--few_shot_greedy_dimension", type=int, default=128)
    parser.add_argument("--few_shot_normalize", type=str, default="True")
    ### calculating distance -> classsize
    parser.add_argument("--voter_distance_autothresh", type=str, default="False")
    parser.add_argument(
        "--voter_distance_thresh", type=float, default=np.cos(np.pi / 4)
    )
    parser.add_argument("--voter_distance_use_mean", type=str, default="False")
    ### capturing classsize -> few shot
    parser.add_argument("--voter_classcount_autothresh", type=str, default="False")
    parser.add_argument("--voter_classcount_thresh", type=int, default=5)
    # 2.3. LoF (FEAS)
    parser.add_argument("--noise_captor", type=str, default="")
    parser.add_argument("--noise_greedy_ratio", type=float, default=1.0)
    parser.add_argument("--noise_greedy_dimension", type=int, default=128)
    parser.add_argument("--noise_normalize", type=str, default="False")
    ### lof configs
    parser.add_argument("--lof_k", type=int, default=6)
    parser.add_argument("--lof_thresh", type=float, default=0.15)
    parser.add_argument("--lof_weight", type=str, default="False")
    ### calculating distance -> classsize
    parser.add_argument("--voter_distance_autothresh_mid", type=str, default="False")
    parser.add_argument(
        "--voter_distance_thresh_mid",
        type=float,
        default=np.cos(np.arccos(1 / (2048**0.5)) / 2),
    )
    parser.add_argument("--voter_distance_use_mean_mid", type=str, default="False")
    ### capturing classsize -> few shot
    parser.add_argument("--voter_classcount_autothresh_mid", type=str, default="False")
    parser.add_argument("--voter_classcount_thresh_mid", type=int, default=5)

    # 3. Eval (faiss configs)
    parser.add_argument("--faiss_distance", type=str, default="L2")  # for faiss scorer
    parser.add_argument("--faiss_k_neighbor", type=int, default=1)
    parser.add_argument("--faiss_on_gpu", type=str, default="True")
    parser.add_argument("--faiss_num_workers", type=int, default=8)

    # 4. Hardware support / Others
    parser.add_argument("--gpu", type=int, default=[0])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generate_coreset_only", type=str, default="False")

    args = parser.parse_args(args=[])

    mvtec_clist, visa_clist, imagenet_clist = utils.clists(args)
    if args.dataset == "mvtec":
        args.subdatasets = mvtec_clist
    elif args.dataset == "visa":
        args.subdatasets = visa_clist
    elif args.dataset == "imagenet_lt":
        args.subdatasets = imagenet_clist
    else:
        raise NotImplementedError()

    args.synthetic_data_path = utils.synthetic_data_path(args)
    args.coreset_dir = utils.coreset_dir(args)
    args.result_dir = utils.result_dir(args)

    return args


def main(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    device = utils.set_torch_device(args.gpu)
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    with open(os.path.join(args.synthetic_data_path, "feas.pickle"), "rb") as f:
        features = pickle.load(f)
    with open(os.path.join(args.synthetic_data_path, "gaps.pickle"), "rb") as f:
        gaps = pickle.load(f)

    assert features.shape[0] % gaps.shape[0] == 0, "file corrupt"
    feamap_size = int(features.shape[0] / gaps.shape[0])

    patchmaker = PatchMaker(patchsize=3, stride=1)

    image_paths_ = sorted(
        glob.glob(os.path.join(args.synthetic_data_path, "*", "train", "good", "*"))
    )
    image_paths = [i for i in image_paths_ if i.endswith(".png")]

    noise_label = []

    for image in image_paths:
        if image.split("/")[-1][:3].isnumeric():
            mask = np.zeros(shape=feamap_size)

        else:
            image = image.replace("SYMLINKS/project_tailcaptor/synthetic", "generic")
            image = os.path.join("/", *image.split("/")[1:7], *image.split("/")[9:])
            image_no = image.split("/")[-1]
            anomaly_type, image_no = image_no[:-8], image_no[-7:]
            image = os.path.join("/", *image.split("/")[1:8])
            image = os.path.join(image, "ground_truth", anomaly_type, image_no)
            image = image[:-4] + "_mask.png"

            if not os.path.exists(image):
                print(image)
                raise ReferenceError()

            # The images go through identical process of resizing and center-cropping
            transform = [
                transforms.Resize(args.resize),
                transforms.CenterCrop(args.imagesize),
                transforms.ToTensor(),
            ]
            transform = transforms.Compose(transform)

            mask = PIL.Image.open(image)
            mask = transform(mask)

            # Substituted by transform (to be deleted)
            # mask = mask.reshape(1, 1, args.imagesize, args.imagesize)

            # This approximates the receptive field of Wideresnet by simply interpolating.
            # More accurate way would be to set weights of the network to 1 and bias to 0, and then passing through the network.
            mask = torch.nn.functional.interpolate(mask, size=(28, 28))

            # The images go through same patchifying algorithm.
            mask = patchmaker.patchify(mask)
            mask = mask.reshape(-1, 1, 3, 3)
            mask = torch.mean(mask, dim=(-1, -2)).squeeze()

            mask = mask.to("cpu").numpy()

        noise_label.append(mask)

    noise_label = np.concatenate(noise_label, axis=0)
    noise_label[noise_label != 0] = 1


if __name__ == "__main__":
    args = parse_args()
    main(args)
