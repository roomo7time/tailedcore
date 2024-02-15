import numpy as np

from src import utils
from src.dataloader import get_dataloaders
from src.backbone import get_backbone
from src.feature_embedder import FeatureEmbedder
from src.coreset_model import get_coreset_model
from src.inferencer import infer_on_image_dataloader, infer_on_patch_dataloader
from src.get_args import parse_args


def evaluate(args):
    utils.set_seed(args.config.seed)

    config = args.config

    if args.data_format in ["mvtec", "mvtec-multiclass"]:
        infer_on_dataloader = infer_on_image_dataloader
    elif args.data_format in ["labelme"]:
        infer_on_dataloader = infer_on_patch_dataloader
    else:
        raise ValueError()

    device = utils.set_torch_device(args.gpu)

    input_shape = (3, config.data.inputsize, config.data.inputsize)

    dataloaders = get_dataloaders(
        config,
        data_format=args.data_format,
        data_path=args.data_path,
        batch_size=args.batch_size,
    )

    for _dataloaders in dataloaders:
        _train_dataloader = _dataloaders["train"]
        _test_dataloader = _dataloaders["test"]

        labels_gt = []
        masks_gt = []

        # move to get_dataloaders part
        if args.data_format in ["mvtec", "mvtec-multiclass"]:
            for data in _test_dataloader:
                labels_gt.extend(data["is_anomaly"].numpy().tolist())
                masks_gt.extend(data["mask"].numpy().tolist())

        elif args.data_format in ["labelme"]:
            labels_gt = list(_test_dataloader.dataset.is_anomaly.values())
            masks_gt = [
                mask_gt.tolist()
                for mask_gt in list(_test_dataloader.dataset.get_masks().values())
            ]
        else:
            raise NotImplementedError()

        for backbone_name in config.model.backbone_names:
            backbone = get_backbone(backbone_name)
            feature_embedder = FeatureEmbedder(
                device, input_shape, backbone, config.model.layers_to_extract
            )

            save_dir_path = f"./artifacts/{args.data_name}/{args.config_name}/{_train_dataloader.name}"

            coreset_model = get_coreset_model(
                config,
                feature_embedder=feature_embedder,
                device=device,
                faiss_on_gpu=args.faiss_on_gpu,
                faiss_num_workers=args.faiss_num_workers,
                sampler_on_gpu=args.sampler_on_gpu,
                save_dir_path=save_dir_path,
            )

            coreset_model.incremental_fit(_train_dataloader)

            del coreset_model


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
