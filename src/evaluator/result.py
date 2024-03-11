import os
import numpy as np
import pandas as pd

from . import th_tuner, metrics, on_blobs
from .. import utils


def save_result(
    image_paths,
    image_scores: np.ndarray,
    labels_gt,
    score_masks: np.ndarray,
    masks_gt,
    save_dir_path,
    num_ths=21,
    min_size=60,
):
    save_plot_dir = os.path.join(save_dir_path, "plot")

    # ths = np.around(
    #     np.linspace(np.min(score_masks), np.max(score_masks), num_ths),
    #     3)[1:-1]

    # th_max_l1_sim = th_tuner.tune_score_threshold(masks_gt,
    #                                               score_masks,
    #                                               score_thresholds=ths,
    #                                               metric_type="l1_sim")

    # th_max_iou = th_tuner.tune_score_threshold(masks_gt,
    #                                            score_masks,
    #                                            score_thresholds=ths,
    #                                            metric_type="iou")

    # th_upper_bound = ths[-1] if th_max_iou <= th_max_iou else th_max_iou

    # th_min_fnfp = th_tuner.tune_score_threshold(
    #     masks_gt,
    #     score_masks,
    #     score_thresholds=ths[(ths >= th_max_l1_sim) & (ths <= th_upper_bound)],
    #     min_size=min_size,
    #     metric_type="fnfp")

    # tp, tn, fp, fn = on_blobs.compute_metrics(
    #     masks_gt=on_blobs.threshold_score_masks(masks_gt, 0.5, min_size),
    #     masks_pred=on_blobs.threshold_score_masks(score_masks, th_min_fnfp,
    #                                               min_size),
    #     iou_threshold=0.0125)

    # print(
    #     f"final metrics - tp: {tp} tn: {tn} fp: {fp} fn {fn} - th: {th_min_fnfp}"
    # )

    # ths_tuned = {
    #     "th_max_iou": th_max_iou,
    #     "th_max_l1_sim": th_max_l1_sim,
    #     "th_min_fnfp": th_min_fnfp,
    # }

    # utils.plot_hist(
    #     score_masks,
    #     masks_gt,
    #     filename=os.path.join(save_plot_dir, "hist_pixel_scores_all.png"),
    #     other_points=ths_tuned,
    # )

    # utils.plot_hist(score_masks[masks_gt == 1],
    #                 filename=os.path.join(save_plot_dir,
    #                                       "hist_pixel_scores_anomaly.png"))

    # ths_raw = {
    #     f"th_{th:.3f}": th
    #     for th in ths[(ths >= th_max_l1_sim) & (ths <= th_max_iou)]
    # }

    # for th_name, th_val in {**ths_raw, **ths_tuned}.items():
    #     utils.plot_score_masks(
    #         save_dir_path=os.path.join(save_plot_dir, f"{th_name}_filtered"),
    #         image_paths=image_paths,
    #         masks_gt=on_blobs.threshold_score_masks(masks_gt, 0.5, min_size),
    #         score_masks=score_masks,
    #         image_scores=image_scores,
    #         binary_masks=on_blobs.threshold_score_masks(
    #             score_masks, th_val, min_size))

    #     utils.plot_score_masks(
    #         save_dir_path=os.path.join(save_plot_dir, f"{th_name}"),
    #         image_paths=image_paths,
    #         masks_gt=masks_gt,
    #         score_masks=score_masks,
    #         image_scores=image_scores,
    #         binary_masks=on_blobs.threshold_score_masks(
    #             score_masks, th_val, None))

    # utils.plot_score_masks(
    #     save_dir_path=os.path.join(save_plot_dir, "scores"),
    #     image_paths=image_paths,
    #     masks_gt=masks_gt,
    #     score_masks=score_masks,
    #     image_scores=image_scores,
    # )
    utils.plot_mvtec_score_masks(
        save_dir_path=os.path.join(save_plot_dir, "scores"),
        image_paths=image_paths,
        masks_gt=masks_gt,
        score_masks=score_masks,
    )

    try:
        print("Computing image auroc...")
        image_auroc = metrics.compute_imagewise_retrieval_metrics(
            image_scores, labels_gt
        )["auroc"]
        print("Computing pixel auroc...")
        pixel_auroc = metrics.compute_pixelwise_retrieval_metrics(
            score_masks, masks_gt
        )["auroc"]
    except:
        image_auroc = 0.0
        pixel_auroc = 0.0
        print("Failed at computing image auroc...")

    result = {"test_data_name": os.path.basename(save_dir_path)}
    result["image_auroc"] = image_auroc * 100
    result["pixel_auroc"] = pixel_auroc * 100

    return result


def summarize_result(result_list, save_dir_path):
    df = pd.DataFrame(result_list)

    # Save to CSV
    save_path = os.path.join(save_dir_path, "result.csv")
    df.to_csv(save_path, index=False)  # 'index=False' to avoid writing row numbers

    return df
