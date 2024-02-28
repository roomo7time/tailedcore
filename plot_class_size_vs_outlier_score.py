import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
import os

import plot_utils
from src.sampler import LOFSampler, TailSampler


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

    extracted, artifact_name = get_extracted_artifacts()

    feas = extracted["feas"]
    gaps = extracted["gaps"]
    rmasks = extracted["downsized_masks"]

    class_sizes = extracted["class_sizes"]

    b, fea_dim, h, w = feas.size()

    patch_class_sizes = class_sizes[:, None].repeat(1, h * w).reshape((-1,)).float()
    is_few_shot_gt = (patch_class_sizes < 20).long()
    is_medium_shot_gt = ((patch_class_sizes >= 20) & (patch_class_sizes < 100)).long()
    is_many_shot_gt = (patch_class_sizes >= 100).long()

    is_anomaly_gt = torch.round(rmasks).reshape((-1,)).long()
    labels = is_few_shot_gt + is_medium_shot_gt * 2 + is_many_shot_gt * 3
    patch_features = (
        feas.reshape(b, fea_dim, -1).permute(0, 2, 1).reshape((-1, fea_dim))
    )

    lof_sampler = LOFSampler()
    tail_sampler = TailSampler()

    _, tail_indices = tail_sampler.run(gaps)

    outlier_scores_path = f"./shared_resources/{artifact_name}_outlier_scores.pt"

    if os.path.exists(outlier_scores_path):
        outlier_scores = torch.load(outlier_scores_path)
    else:
        _, _, outlier_scores = lof_sampler.run(
            patch_features, feature_map_shape=(h, w), return_outlier_scores=True
        )
        torch.save(outlier_scores, outlier_scores_path)

    labels[is_anomaly_gt.bool()] = 0

    index_map = torch.arange(len(outlier_scores)).reshape(-1, h*w)
    ones = torch.ones((len(gaps), )).bool()
    ones[tail_indices] = False
    indices_head = index_map[ones].reshape((-1))

    # outlier_scores_few_shot = outlier_scores[shot_labels == 1]
    # outlier_scores_medium_shot = outlier_scores[shot_labels == 2]
    # outlier_scores_many_shot = outlier_scores[shot_labels == 3]

    # class_sizes_few_shot = patch_class_sizes[shot_labels == 1]
    # class_sizes_medium_shot = patch_class_sizes[shot_labels == 2]
    # class_sizes_many_shot = patch_class_sizes[shot_labels == 3]

    # mean_outlier_score_few_shot = outlier_scores_few_shot.mean().item()
    # mean_outlier_score_medium_shot = outlier_scores_medium_shot.mean().item()
    # mean_outlier_score_many_shot = outlier_scores_many_shot.mean().item()

    # mean_class_size_few_shot = class_sizes_few_shot.mean().item()
    # mean_class_size_medium_shot = class_sizes_medium_shot.mean().item()
    # mean_class_size_many_shot = class_sizes_many_shot.mean().item()

    # means = np.array(
    #     [
    #         [mean_outlier_score_few_shot, mean_class_size_few_shot],
    #         [mean_outlier_score_medium_shot, mean_class_size_medium_shot],
    #         [mean_outlier_score_many_shot, mean_class_size_many_shot],
    #     ]
    # )

    """
    score
    """    
    def calculate_label_counts(scores, labels, p=0.85, num_labels=4):
        
        thresh = np.quantile(scores, p)

        labels_sampled = labels[scores < thresh]

        label_counts =  [np.sum(labels_sampled == i) for i in range(num_labels)]
        
        return label_counts
    
    outlier_scores_baseline = outlier_scores.numpy()
    labels_baseline = labels.numpy()

    coreset_ratio = 0.1
    
    labels_counts_source = calculate_label_counts(outlier_scores_baseline, labels_baseline, p=1.0)
    label_counts_baseline = calculate_label_counts(outlier_scores_baseline, labels_baseline)
    label_counts_ours = list(label_counts_baseline)
    label_counts_ours[1] += h*w*len(tail_indices)

    # label_counts_baseline = np.log(label_counts_baseline)
    # label_counts_ours = np.log(label_counts_ours)

    """
    plots
    """
    label_names = ["Few-shot", "Medium-shot", "Many-shot"]

    # plot_utils.plot_bars(
    #     list1=label_counts_baseline[1:],
    #     list2=label_counts_ours[1:],
    #     list1_name="baseline",
    #     list2_name="ours",
    #     x_ticks=label_names,
    #     y_label="Ratio of lowest density",
    #     filename='./test.jpg',
    #     show=True
    # )

    plot_utils.plot_bars_three_lists(
        list1=labels_counts_source[1:],
        list2=label_counts_baseline[1:],
        list3=label_counts_ours[1:],
        list1_name="before noise discrimination",
        list2_name="baseline",
        list3_name="ours",
        x_ticks=label_names,
        y_label="Ratio of lowest density",
        filename='./test.jpg',
        show=True
    )


vivid_colors = [
    "#FF6B6B",  # Vivid Red 0
    "#FFD93D",  # Vivid Yellow 1
    "#6BCB77",  # Vivid Green 2
    "#4D96FF",  # Vivid Blue 3
    "#9D4EDD",  # Vivid Purple 4
    "#FF70A6",  # Vivid Pink 5
    "#FF9F45",  # Vivid Orange 6
    "#32E0C4",  # Vivid Teal 7
    "#F06595",  # Vivid Magenta
    "#A9DEF9",  # Soft Blue
    "#E4C1F9",  # Soft Purple
    "#FEC8D8",  # Soft Pink
    "#FAFCC2",  # Soft Yellow
]


def plot_top_outlier_scores_ratio(outlier_scores, labels):
    # Define label names
    label_names = ["Anomaly", "Few-shot", "Medium-shot", "Many-shot"]

    # Calculate the cutoff for the top 15%
    cutoff_index = int(len(outlier_scores) * 0.15)  # Get the index for the cutoff
    top_indices = np.argsort(outlier_scores)[
        -cutoff_index:
    ]  # Indices of the top 15% scores

    # Extract the labels for the top 15%
    top_labels = labels[top_indices]

    # Calculate the ratio of each label in the top 15%
    label_ratios = [np.mean(top_labels == i) for i in range(4)]

    plot_utils.plot_bars(
        list1=label_ratios,
        list2=label_ratios,
        list1_name="baseline",
        list2_name="ours",
        x_ticks=label_names,
        y_label="Ratio of lowest density",
        filename='./test.jpg'
    )



def plot_label_vs_outlier_score(outlier_scores, labels, filename="plot.png"):
    fig, ax = plt.subplots()

    # Set the scatter plot
    scatter = ax.scatter(outlier_scores, labels, alpha=0.5)

    ax.set_xlabel("Outlier Score")
    ax.set_ylabel("Label")

    # Custom labels for y-axis
    label_names = {0: "Anomaly", 1: "Few-shot", 2: "Medium-shot", 3: "Many-shot"}
    unique_labels = sorted(set(labels))
    plt.yticks(unique_labels, [label_names[label] for label in unique_labels])

    plt.title("Label vs Outlier Score")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(filename, dpi=300)  # Save as a high-resolution image
    plt.close()  # Close the plot to free memory


def plot_class_size_vs_outlier_score(class_sizes, outlier_scores, labels, means):
    fig, ax = plt.subplots()

    # Import color palettes
    from palettable.colorbrewer.qualitative import Pastel1_7, Set1_9

    # Combine palettes and select colors
    palette = Pastel1_7.hex_colors + Set1_9.hex_colors
    # Colors for anomaly, few-shot, medium-shot, and many-shot
    colors = [
        palette[4],
        palette[0],
        palette[1],
        palette[2],
    ]  # Reordered for the new labeling

    # Common marker for few-shot, medium-shot, and many-shot; distinct for anomaly
    marker = "o"  # Circle marker for all categories
    anomaly_marker = "X"  # Distinct marker for anomaly

    # Loop through labels and plot each subset of data
    for label in range(4):  # 0 for anomaly, 1, 2, 3 for shot types
        idx = np.where(labels == label)
        ax.scatter(
            class_sizes[idx],
            outlier_scores[idx],
            c=colors[label],
            marker=anomaly_marker if label == 0 else marker,
            label=f'{["Anomaly", "Few", "Medium", "Many"][label]}',
            alpha=1 if label == 0 else 0.05,  # Make anomaly points fully opaque
        )

    # Plot means for Few-shot, Medium-shot, Many-shot; exclude Anomaly from means
    mean_markers = "D"  # Distinct marker for mean points
    for i, (mean_outlier_score, mean_class_size) in enumerate(
        means[1:]
    ):  # Start from 1 to exclude anomaly
        ax.scatter(
            mean_class_size,
            mean_outlier_score,
            c=colors[i + 1],  # +1 because means[0] is excluded
            marker=mean_markers,
            edgecolor="black",
            linewidth=2,
            s=100,
            label=f'{["Few", "Medium", "Many"][i]}-shot Mean',
        )

    # Legend
    leg = plt.legend(
        loc="upper right",
        title="Category",
    )
    for lh in leg.legendHandles:
        lh.set_alpha(1)

    plt.xlabel("Class Size")
    plt.ylabel("Outlier Score")
    plt.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0)
    dpi = 300
    plt.savefig("./plot_adjusted_for_anomaly_as_0.png", dpi=dpi)
    plt.close()


if __name__ == "__main__":

    plots()
