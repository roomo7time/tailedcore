import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
import os
import torch.nn.functional as F
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import KernelDensity

import plot_utils
from src.sampler import LOFSampler, TailSampler, GreedyCoresetSampler
from src.utils import set_seed


def get_extracted_artifacts():

    extracted, artifact_name = (
        _get_extracted_artifacts_mvtec_step_nr10_tk4_seed0_wrn50()
    )

    return extracted, artifact_name


def _get_extracted_artifacts_mvtec_step_nr10_tk4_seed0_wrn50():

    artifact_name = "extracted_mvtec_step_nr10_tk4_tr60_seed0_wrn50"

    extracted_path = f"./shared_resources/{artifact_name}.pt"

    extracted = torch.load(extracted_path)

    return extracted, artifact_name


def plots():

    set_seed(0)
    extracted, artifact_name = get_extracted_artifacts()

    feas = extracted["feas"]
    embeddings = extracted["gaps"]
    rmasks = extracted["downsized_masks"]
    class_names = extracted["class_names"]
    class_sizes = extracted["class_sizes"]

    b, fea_dim, h, w = feas.size()

    patch_class_sizes = class_sizes[:, None].repeat(1, h * w).reshape((-1,)).float()
    is_few_shot_gt = (patch_class_sizes < 20).long()
    is_medium_shot_gt = ((patch_class_sizes >= 20) & (patch_class_sizes < 100)).long()
    is_many_shot_gt = (patch_class_sizes >= 100).long()

    is_anomaly_gt = torch.round(rmasks).reshape((-1,)).long()
    shot_labels = is_few_shot_gt + is_medium_shot_gt * 2 + is_many_shot_gt * 3
    patch_features = (
        feas.reshape(b, fea_dim, -1).permute(0, 2, 1).reshape((-1, fea_dim))
    )

    lof_sampler = LOFSampler()
    tail_sampler = TailSampler()

    _, tail_indices, cizes = tail_sampler.run(embeddings, return_class_sizes=True)

    outlier_scores_path = f"./shared_resources/{artifact_name}_outlier_scores.pt"

    if os.path.exists(outlier_scores_path):
        outlier_scores = torch.load(outlier_scores_path)
    else:
        _, _, outlier_scores = lof_sampler.run(
            patch_features, feature_map_shape=(h, w), return_outlier_scores=True
        )
        torch.save(outlier_scores, outlier_scores_path)

    shot_labels[is_anomaly_gt.bool()] = 0

    # extra scores
    kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(embeddings)
    log_densities = kde.score_samples(embeddings)
    kdenlds = -log_densities

    clf = LocalOutlierFactor(n_neighbors=6, metric="l2")
    clf.fit(embeddings)
    lofs = -clf.negative_outlier_factor_

    cizes = cizes[:, None].repeat(1, h * w).reshape((-1,)).numpy()
    lofs = torch.from_numpy(lofs[:, None]).repeat(1, h * w).reshape((-1,)).numpy()
    kdenlds = torch.from_numpy(kdenlds[:, None]).repeat(1, h * w).reshape((-1,)).numpy()

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))  # Adjust the figsize as needed

    _plot_removal_shot_ratio(outlier_scores, shot_labels, ax=axs[0])
    _plot_density_patch_vs_embedding(
        outlier_scores,
        (h, w),
        shot_labels,
        embeddings,
        ax=axs[1]
    )

    filename = './dillema.jpg'
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches="tight", format=os.path.splitext(filename)[1][1:], dpi=300)
    plt.close()

# def _plot_tpr_vs_fpr(
#     patch_lofs,
#     lofs,
#     neg_log_densities,
#     cizes,
#     few_shot_labels,
# ):
#     plot_utils.plot_roc_curves(
#         [patch_lofs, lofs, neg_log_densities, cizes],
#         true_labels=few_shot_labels,
#         score_names=["LOF-patch", "LOF", "KDE", "Ours"],
#     )


def _plot_density_patch_vs_embedding(
    feature_outlier_scores,
    feature_map_shape,
    shot_labels,
    embeddings,
    ax=None,
):
    h, w = feature_map_shape

    embeddings = F.normalize(embeddings, dim=-1).numpy()

    kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(embeddings)
    log_density = kde.score_samples(embeddings)

    embedding_outlier_scores = -log_density
    # embedding_outlier_scores = - np.exp(log_density)

    # clf = LocalOutlierFactor(n_neighbors=6, metric="l2")
    # clf.fit(embeddings)
    # embedding_outlier_scores = -clf.negative_outlier_factor_

    n = len(feature_outlier_scores)

    embedding_outlier_scores = (
        torch.from_numpy(embedding_outlier_scores)[:, None]
        .repeat(1, h * w)
        .reshape((-1,))
    )

    # random_indices = np.random.choice(n, size=int(n*0.01), replace=False)
    # feature_outlier_scores = feature_outlier_scores[random_indices]
    # embedding_outlier_scores = embedding_outlier_scores[random_indices]

    plot_utils.plot_and_save_correlation_graph(
        scores1=embedding_outlier_scores.numpy(),
        scores2=feature_outlier_scores.numpy(),
        labels=shot_labels,
        label_names=["Anomaly", "Few-shot", "Medium-shot", "Many-shot"],
        score1_name="Negative log density of embedding",
        score2_name="Patch outlier score",
        ylim=(0.9, 2.5),
        filename=None, #'./density_patch_vs_embedding.jpg'
        ax=ax,
    )


def _plot_removal_shot_ratio(outlier_scores, labels, ax=None):
    def calculate_label_counts(scores, labels, p=0.85, num_labels=4):

        thresh = np.quantile(scores, p)

        labels_sampled = labels[scores > thresh]

        label_counts = [np.sum(labels_sampled == i) for i in range(num_labels)]

        return label_counts

    labels_counts = calculate_label_counts(
        outlier_scores.numpy(), labels.numpy(), p=0.85
    )

    removal_ratio = [[]] * len(labels_counts)
    for i in range(len(labels_counts)):
        removal_ratio[i] = labels_counts[i] / (labels == i).sum().item()

    label_names = ["Anomaly", "Few-shot", "Medium-shot", "Many-shot"]

    plot_utils.plot_bars(
        data_lists=[removal_ratio],
        list_names=["none"],
        x_ticks=label_names,
        y_label="Ratio",
        legend=False,
        filename=None, #"./removal_ratio.jpg",
        ax=ax
    )


# def plot_top_outlier_scores_ratio(outlier_scores, labels):
#     # Define label names
#     label_names = ["Anomaly", "Few-shot", "Medium-shot", "Many-shot"]

#     # Calculate the cutoff for the top 15%
#     cutoff_index = int(len(outlier_scores) * 0.15)  # Get the index for the cutoff
#     top_indices = np.argsort(outlier_scores)[
#         -cutoff_index:
#     ]  # Indices of the top 15% scores

#     # Extract the labels for the top 15%
#     top_labels = labels[top_indices]

#     # Calculate the ratio of each label in the top 15%
#     label_ratios = [np.mean(top_labels == i) for i in range(4)]

#     plot_utils.plot_bars(
#         list1=label_ratios,
#         list2=label_ratios,
#         list1_name="baseline",
#         list2_name="ours",
#         x_ticks=label_names,
#         y_label="Ratio of lowest density",
#         filename="./test.jpg",
#     )


# def plot_label_vs_outlier_score(outlier_scores, labels, filename="plot.png"):
#     fig, ax = plt.subplots()

#     # Set the scatter plot
#     scatter = ax.scatter(outlier_scores, labels, alpha=0.5)

#     ax.set_xlabel("Outlier Score")
#     ax.set_ylabel("Label")

#     # Custom labels for y-axis
#     label_names = {0: "Anomaly", 1: "Few-shot", 2: "Medium-shot", 3: "Many-shot"}
#     unique_labels = sorted(set(labels))
#     plt.yticks(unique_labels, [label_names[label] for label in unique_labels])

#     plt.title("Label vs Outlier Score")
#     plt.grid(True, which="both", linestyle="--", linewidth=0.5)
#     plt.tight_layout()

#     # Save the plot to a file
#     plt.savefig(filename, dpi=300)  # Save as a high-resolution image
#     plt.close()  # Close the plot to free memory


# def plot_class_size_vs_outlier_score(class_sizes, outlier_scores, labels, means):
#     fig, ax = plt.subplots()

#     # Import color palettes
#     from palettable.colorbrewer.qualitative import Pastel1_7, Set1_9

#     # Combine palettes and select colors
#     palette = Pastel1_7.hex_colors + Set1_9.hex_colors
#     # Colors for anomaly, few-shot, medium-shot, and many-shot
#     colors = [
#         palette[4],
#         palette[0],
#         palette[1],
#         palette[2],
#     ]  # Reordered for the new labeling

#     # Common marker for few-shot, medium-shot, and many-shot; distinct for anomaly
#     marker = "o"  # Circle marker for all categories
#     anomaly_marker = "X"  # Distinct marker for anomaly

#     # Loop through labels and plot each subset of data
#     for label in range(4):  # 0 for anomaly, 1, 2, 3 for shot types
#         idx = np.where(labels == label)
#         ax.scatter(
#             class_sizes[idx],
#             outlier_scores[idx],
#             c=colors[label],
#             marker=anomaly_marker if label == 0 else marker,
#             label=f'{["Anomaly", "Few", "Medium", "Many"][label]}',
#             alpha=1 if label == 0 else 0.05,  # Make anomaly points fully opaque
#         )

#     # Plot means for Few-shot, Medium-shot, Many-shot; exclude Anomaly from means
#     mean_markers = "D"  # Distinct marker for mean points
#     for i, (mean_outlier_score, mean_class_size) in enumerate(
#         means[1:]
#     ):  # Start from 1 to exclude anomaly
#         ax.scatter(
#             mean_class_size,
#             mean_outlier_score,
#             c=colors[i + 1],  # +1 because means[0] is excluded
#             marker=mean_markers,
#             edgecolor="black",
#             linewidth=2,
#             s=100,
#             label=f'{["Few", "Medium", "Many"][i]}-shot Mean',
#         )

#     # Legend
#     leg = plt.legend(
#         loc="upper right",
#         title="Category",
#     )
#     for lh in leg.legendHandles:
#         lh.set_alpha(1)

#     plt.xlabel("Class Size")
#     plt.ylabel("Outlier Score")
#     plt.tight_layout()
#     fig.subplots_adjust(hspace=0, wspace=0)
#     dpi = 300
#     plt.savefig("./plot_adjusted_for_anomaly_as_0.png", dpi=dpi)
#     plt.close()


if __name__ == "__main__":

    plots()
