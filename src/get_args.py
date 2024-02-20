import os
import argparse
from tabulate import tabulate

from .utils import load_config_args


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root_path",
        type=str,
        default="/home/jay/mnt/hdd01/data/image_datasets",
    )

    # few-shot setting
    parser.add_argument(
        "--data_sub_path",
        type=str,
        default="anomaly_detection/symlink_mvtec_step_nr10_k4_seed1",
        # default="anomaly_detection/symlink_mvtec_step_nr10_k1_seed0",
        # default="anomaly_detection/symlink_mvtec_pareto_nr10_seed0",
    )
    parser.add_argument("--data_format", type=str, default="mvtec-multiclass")
    parser.add_argument("--config_name", type=str, default="tailedpatch_mvtec_01")

    # # toy setting
    # parser.add_argument(
    #     "--data_sub_path",
    #     type=str,
    #     default=
    #     "anomaly_detection/mvtec_anomaly_detection_toy",
    # )
    # parser.add_argument("--data_format", type=str,
    #                     default="mvtec-multiclass")
    # parser.add_argument("--config_name",
    #                     type=str,
    #                     default="patchcore_mvtec_toy_01")

    # parser.add_argument(
    #     "--data_sub_path",
    #     type=str,
    #     default="anomaly_detection/mvtec_anomaly_detection",
    # )
    # parser.add_argument("--data_format", type=str,
    #                     default="mvtec-multiclass")  # labelme, mvtec
    # parser.add_argument("--config_name", type=str, default="patchcore_mvtec_01")

    # parser.add_argument(
    #     "--data_sub_path",
    #     type=str,
    #     default="hankook_tire/ml/toy/version01",
    # )
    # parser.add_argument("--data_format", type=str, default="labelme")
    # parser.add_argument("--config_name", type=str, default="patchcore_toy_03")

    # parser.add_argument(
    #     "--data_sub_path",
    #     type=str,
    #     default="hankook_tire/ml/RF10/version03",
    # )
    # parser.add_argument("--data_format", type=str, default="labelme")
    # parser.add_argument("--config_name", type=str, default="patchcore_RF10_08")

    parser.add_argument("--gpu", type=int, default=[0])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--faiss_on_gpu", type=bool, default=True)
    parser.add_argument("--faiss_num_workers", type=int, default=0)
    parser.add_argument("--sampler_on_gpu", type=bool, default=True)
    #####################################################################

    args = parser.parse_args()

    args.config = load_config_args(
        os.path.join("./configs", args.config_name + ".yaml")
    )

    args.data_path = os.path.join(args.data_root_path, args.data_sub_path)
    args.data_name = "_".join(
        [args.data_sub_path.lstrip(os.sep).replace(os.sep, "_"), args.data_format]
    )

    if args.data_format in ["mvtec", "mvtec-multiclass"]:
        args.patch_infer = False
    elif args.data_format in ["labelme"]:
        args.patch_infer = True
    else:
        raise ValueError()

    print(tabulate(list(vars(args).items()), headers=["arguments", "values"]))

    return args
