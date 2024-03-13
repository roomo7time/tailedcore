import os
import numpy as np

import src.evaluator.result as result

from src import utils
from src.dataloader import get_dataloaders
from src.get_args import parse_args
from src.engine import AblationEngine


def evaluate(args):
    utils.set_seed(args.config.seed)

    config = args.config

    device = utils.set_torch_device(args.gpu)

    input_shape = (3, config.data.inputsize, config.data.inputsize)

    dataloaders = get_dataloaders(
        config.data,
        data_format=args.data_format,
        data_path=args.data_path,
        batch_size=args.batch_size,
    )

    result_list = []

    for _dataloaders in dataloaders:
        _train_dataloader = _dataloaders["train"]
        _test_dataloader = _dataloaders["test"]

        save_train_dir_path = os.path.join(
            "./artifacts", args.data_name, args.config_name, _train_dataloader.name
        )
        save_test_dir_path = os.path.join(save_train_dir_path, _test_dataloader.name)
        save_outputs_path = os.path.join(save_test_dir_path, "outputs.pkl")

        if os.path.exists(save_outputs_path):
            outputs = utils.load_dict(save_outputs_path)
            image_scores = outputs["image_scores"]
            score_masks = outputs["score_masks"]

            labels_gt = outputs["labels_gt"]
            masks_gt = outputs["masks_gt"]
            image_paths = outputs["image_paths"]
            del outputs
        else:

            labels_gt = []
            masks_gt = []

            # move to get_dataloaders part
            if args.data_format in ["mvtec", "mvtec-multiclass"]:
                for data in _test_dataloader:
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    masks_gt.extend(data["mask"].numpy().tolist())

            elif args.data_format in ["labelme"]:
                labels_gt = np.array(list(_test_dataloader.dataset.is_anomaly.values()))
                masks_gt = np.array(
                    [
                        np.array(mask_gt).astype(np.uint8)
                        for mask_gt in list(
                            _test_dataloader.dataset.get_masks().values()
                        )
                    ]
                ).squeeze()
            else:
                raise NotImplementedError()

            image_scores = []
            score_masks = []

            for backbone_name in config.model.backbone_names:

                engine = AblationEngine(
                    config=config,
                    backbone_name=backbone_name,
                    device=device,
                    input_shape=input_shape,
                    train_dataloader=_train_dataloader,
                    test_dataloader=_test_dataloader,
                    faiss_on_gpu=args.faiss_on_gpu,
                    faiss_num_workers=args.faiss_num_workers,
                    sampler_on_gpu=args.sampler_on_gpu,
                    save_dir_path=save_train_dir_path,
                    patch_infer=args.patch_infer,
                    train_mode=getattr(config.model, "train_mode", None),
                )

                engine.train()

                # FIXME: For truely large-scale experiment, image_socre and score mask needs to be saved for each image separately, and tested accordingly in a separate manner.
                (
                    image_scores_per_backbone,
                    score_masks_per_backbone,
                    _image_paths,
                ) = engine.infer()

                image_scores.append(image_scores_per_backbone)
                score_masks.append(score_masks_per_backbone)

                del engine

            image_scores = np.array(image_scores)
            score_masks = np.array(score_masks)
            image_paths = _image_paths

            outputs = {
                "image_scores": image_scores,
                "score_masks": score_masks,
                "labels_gt": labels_gt,
                "masks_gt": masks_gt,
                "image_paths": image_paths,
            }

            utils.save_dict(outputs, save_outputs_path)

        image_scores = utils.minmax_normalize_image_scores(
            image_scores
        )  # this part incldues ensembling of different backbone outputs
        score_masks = utils.minmax_normalize_score_masks(
            score_masks
        )  # this part incldues ensembling of different backbone outputs

        
        masks_gt = np.array(masks_gt).astype(np.uint8)[:, 0, :, :]
        score_masks = np.array(score_masks)
        # FIXME: min_size is currently hard-coded
        result_list.append(
            result.save_result(
                image_paths,
                image_scores,
                labels_gt,
                score_masks,
                masks_gt,
                save_test_dir_path,
                num_ths=41,  # 41
            )
        )

        print(result_list[-1])

    save_log_path = os.path.join("./logs", f'performance_{args.data_name}_{args.config_name}.csv')
    result_df = utils.save_dicts_to_csv(result_list, save_log_path)

    performance = result_df["image_auroc"].mean()
    return performance


if __name__ == "__main__":
    args = parse_args()
    performance = evaluate(args)
    print(f"performance: {performance}")
