"""
For research only
"""

import os
import torch
import math
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import torch.nn as nn

from collections import Counter
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import wasserstein_distance

import src.evaluator.result as result
import src.class_size as class_size
import src.adaptive_class_size as adaptive_class_size
import src.helpers.cv2plot as cv2plot

from src import utils
from src.dataloader import get_dataloaders
from src.get_args import parse_args  # FIXME: make independent args
from src.engine import AblationEngine
from src.backbone import get_backbone
from src.feature_embedder import FeatureEmbedder

from src.patch_maker import PatchMaker
from src.sampler import LOFSampler, TailSampler, TailedLOFSampler, AdaptiveTailSampler







def analyze_gap(args):

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

    train_dataloader = dataloaders[0]["train"]

    backbone_name = config.model.backbone_names[0]

    save_train_dir_path = os.path.join(
        "./artifacts", args.data_name, args.config_name, train_dataloader.name
    )

    ablation_engine = AblationEngine(
        config=config,
        backbone_name=backbone_name,
        device=device,
        input_shape=input_shape,
        train_dataloader=train_dataloader,
        test_dataloader=None,
        faiss_on_gpu=args.faiss_on_gpu,
        faiss_num_workers=args.faiss_num_workers,
        sampler_on_gpu=args.sampler_on_gpu,
        save_dir_path=save_train_dir_path,
        patch_infer=args.patch_infer,
        train_mode=getattr(config.model, "train_mode", None),
    )

    embedding_extractor = ablation_engine.set_embedding_extractor(iter=config.model.embedding_extractor_iter)
    embedding_extractor.fc = nn.Identity()

    _, names_to_ints = ablation_engine._get_dataset_info()

    labels = [[]]*len(train_dataloader)
    gaps = [[]]*len(train_dataloader)

    for data in train_dataloader:

        images = data["image"].to(device)
        label_names = data["classname"]
        _labels = names_to_ints(label_names)
        _gaps = embedding_extractor(images.to(device)).cpu()
    







    return 

    # masks = extracted["masks"]
    labels = extracted["labels"]

    gaps = extracted["gaps"]
    class_names = extracted["class_names"]
    class_sizes = extracted["class_sizes"]

    num_samples_per_class = dict(Counter(class_names))

    save_log_dir = os.path.join("./logs", f"{data_name}_{config_name}")

    save_data_info_path = os.path.join(save_log_dir, "num_samples_per_class.csv")
    utils.save_dicts_to_csv([num_samples_per_class], save_data_info_path)

    num_classes = len(set(class_names))
    save_plot_dir = os.path.join("./artifacts", data_name, config_name)
    print(f"save_plot_dir: {save_plot_dir}")
    plot_gap_analysis(gaps, labels, class_names, class_sizes, save_plot_dir)




def get_relation_matrix(class_labels):
    # Convert class_labels to a numpy array if it isn't already one
    class_labels = np.array(class_labels)
    # Reshape class_labels to enable broadcasting, (n, 1) and (1, n)
    labels_row = class_labels.reshape(-1, 1)
    labels_col = class_labels.reshape(1, -1)
    # Use broadcasting to compare every label with every other label efficiently
    relationship_matrix = (labels_row == labels_col).astype(int)
    
    return relationship_matrix


def _plot_histograms_pmf(scores_same, scores_diff, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Compute Wasserstein distance between the two distributions
    w_dist = wasserstein_distance(scores_same, 1 - scores_diff)
    
    # Plot first score array with specified colors and outline
    plt.hist(scores_same, bins='auto', density=True, alpha=0.65, color='orange', label='Same class')
    # Plot second score array with specified colors and outline
    plt.hist(scores_diff, bins='auto', density=True, alpha=0.65, color='cornflowerblue', label='Different classes')
    
    plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2)
    
    plt.ylabel('Probability mass', size=18)
    plt.xlabel('Normalized arccos of similarity', size=18)
    plt.title(f'Wasserstein Distance: {w_dist:.2f}', size=18)
    plt.legend(fontsize=18, loc="upper left")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)  # Save the figure to the specified file
    plt.close()  # Close the figure to prevent it from displaying in an interactive environment





def _convert_class_names_to_labels(class_names):
    label_map = {}
    class_labels = []
    current_label = 0

    for class_name in class_names:
        if class_name not in label_map:
            label_map[class_name] = current_label
            current_label += 1
        class_labels.append(label_map[class_name])

    class_label_names = list(label_map.keys())

    return torch.LongTensor(class_labels), class_label_names


def convert_indices_to_bool(n: int, indices: torch.Tensor) -> torch.Tensor:
    bool_array = torch.zeros((n), dtype=torch.long)
    bool_array[indices] = 1
    return bool_array






def plot_gap_analysis(
    gaps, labels, class_names, class_sizes, save_dir
):

    if gaps.ndim == 4:
        gaps = gaps[:, :, 0, 0]

    class_labels, class_label_names = _convert_class_names_to_labels(class_names)
    class_labels = class_labels.numpy()

    is_anomaly_gt = labels

    _anomaly_labels = is_anomaly_gt.to(torch.bool).tolist()
    _few_shot_labels = (class_sizes < 20).tolist()
    save_plot_dir = os.path.join(save_dir, "plot_gap")

    self_sim = class_size.compute_self_sim(gaps).numpy()
    relation_matrix = get_relation_matrix(class_labels)
    
    angles_same = []
    angles_diff = []

    for i in range(len(self_sim)):
        _sims = np.clip(self_sim[i], -1, 1)
        _angles = np.arccos(_sims)
        _rel_labels = np.array(relation_matrix[i], dtype=np.bool_)
        _max_angle = _angles.max()
        _angles_same = _angles[_rel_labels] / _max_angle
        _angles_diff = _angles[~_rel_labels] / _max_angle

        angles_same += _angles_same.tolist()
        angles_diff += _angles_diff.tolist()
    angles_same = np.array(angles_same)
    angles_diff = np.array(angles_diff)
        
    _plot_histograms_pmf(angles_same, angles_diff, filename=os.path.join(save_dir, f"hist_{save_dir.split('/')[2]}.pdf"))


import pandas as pd
from typing import List


import pandas as pd
import numpy as np
from typing import List


def average_dfs(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    combined_df = pd.concat(dfs)

    # Identify numeric columns
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns

    # Compute the mean and std only for numeric columns
    avg_numeric_df = combined_df[numeric_cols].groupby(combined_df.index).mean()
    std_numeric_df = combined_df[numeric_cols].groupby(combined_df.index).std()

    # Rename columns in std_numeric_df to indicate they are standard deviations
    std_numeric_df = std_numeric_df.add_suffix("_std")

    # Extract non-numeric columns from the first dataframe in the list
    non_numeric_df = dfs[0][combined_df.select_dtypes(exclude=[np.number]).columns]

    # Concatenate non-numeric, average, and std dataframes
    avg_df = pd.concat([non_numeric_df, avg_numeric_df, std_numeric_df], axis=1)

    return avg_df


def analyze(data="mvtec_all", config_name="extract_mvtec_01", type="gap", seeds: list=list(range(101,106))):

    data_names = get_data_names(data, seeds=seeds,)


    dfs = []

    for data_name in data_names:
        print(f"config: {config_name} data: {data_name}")
        extracted_path = f"./artifacts/{data_name}_mvtec-multiclass/{config_name}/extracted_train_all.pt"
        
        if type == "gap":
            _df = analyze_gap(
                extracted_path=extracted_path,
                data_name=data_name,
                config_name=config_name,
            )
        # elif type == "patch":
        #     _df = analyze_patch(
        #         extracted_path=extracted_path,
        #         data_name=data_name,
        #         config_name=config_name,
        #     )
        else:
            raise NotImplementedError()


def get_data_names(data: str, seeds: list):

    mvtec_data_base_names = [
        "mvtec_step_random_nr10_tk1_tr60",
        "mvtec_step_random_nr10_tk4_tr60",
        "mvtec_pareto_random_nr10",
    ]

    visa_data_base_names = [
        "visa_step_random_nr05_tk1_tr60",
        "visa_step_random_nr05_tk4_tr60",
        "visa_pareto_random_nr05",
    ]

    if data == "mvtec_all":
        data_base_names = mvtec_data_base_names
    elif data == "mvtec_step_tk1":
        data_base_names = [mvtec_data_base_names[0]]
    elif data == "mvtec_step_tk4":
        data_base_names = [mvtec_data_base_names[1]]
    elif data == "mvtec_pareto":
        data_base_names = [mvtec_data_base_names[2]]
    elif data == "visa_all":
        data_base_names = visa_data_base_names
    elif data == "visa_step_tk1":
        data_base_names = [visa_data_base_names[0]]
    elif data == "visa_step_tk4":
        data_base_names = [visa_data_base_names[1]]
    elif data == "visa_pareto":
        data_base_names = [visa_data_base_names[2]]
    elif data == "all":
        data_base_names = mvtec_data_base_names + visa_data_base_names
    else:
        raise NotImplementedError()

    data_names = [
        f"{data_base_name}_seed{seed}"
        for data_base_name in data_base_names
        for seed in seeds
    ]
    
    return data_names


# mvtec:
if __name__ == "__main__":

    args = parse_args()
    analyze_gap(args)
    
