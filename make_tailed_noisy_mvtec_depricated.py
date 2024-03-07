import os
import random
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pareto
from copy import deepcopy
from collections import defaultdict

from src.utils import set_seed, modify_subfolders_in_path, save_dict, load_dict, save_dicts_to_csv

# _MVTEC_CLASS_LIST = [
#     "bottle",
#     "cable",
#     "capsule",
#     "carpet",
#     "grid",
#     "hazelnut",
#     "leather",
#     "metal_nut",
#     "pill",
#     "screw",
#     "tile",
#     "toothbrush",
#     "transistor",
#     "wood",
#     "zipper",
# ]

_DATA_CONFIG_ROOT = './data_configs/mvtec'


def _load_data_config(data_config_path):

    data = load_dict(data_config_path)

    files = data['files']
    train_files = data['train_files']
    anomaly_files = data['anomaly_files']
    num_tail_samples = data['num_tail_samples']
    num_noise_samples = data['num_noise_samples']
    head_classes = data['head_classes']

    return files, train_files, anomaly_files, num_tail_samples, num_noise_samples, head_classes

def _save_data_config(files, train_files, anomaly_files, num_tail_samples, num_noise_samples, head_classes, data_config_path):
    data = {
        "files": files,
        "train_files": train_files,
        "anomaly_files": anomaly_files,
        "num_tail_samples": num_tail_samples,
        "num_noise_samples": num_noise_samples,
        "head_classes": head_classes,
    }

    save_dict(data, data_config_path)
    save_dicts_to_csv([{
        "num_tail_samples": num_tail_samples,
        "num_noise_samples": num_noise_samples,
    }], filename=os.path.splitext(data_config_path)[0] + '.csv')


def make_data_step(
    source_dir: str,
    target_dir: str,
    noise_ratio: float = 0.1,
    noise_on_tail: bool = False,
    tail_k: int = 4,
    tail_class_ratio: float = 0.6,
    seed: int=0,
) -> None:
    set_seed(seed)
    data_config_name = os.path.join(_DATA_CONFIG_ROOT, f'step_nr{int(noise_ratio*100):02d}_k{tail_k}_seed{seed}')

    if noise_on_tail:
        data_config_name += f"_tailnoised"
    
    data_config_path = f"{data_config_name}.pkl"
    # data_config_path = data_config_path.replace("step_", "mvtec_step_")
    # data_config_path = data_config_path.replace("nr10_k", "nr10_tk")
    # data_config_path = data_config_path.replace("_seed", "_tr60_seed")
    

    if os.path.exists(data_config_path):
        files, train_files, anomaly_files, num_tail_samples, num_noise_samples, head_classes = _load_data_config(data_config_path)
    else:

        raise FileNotFoundError()

        # _MVTEC_CLASS_LIST = get_subdirectories(source_dir)

        # class_list = _MVTEC_CLASS_LIST
        # files, train_files, anomaly_files = _get_mvtec_base_file_info(source_dir)
        # num_tail_samples, num_noise_samples, head_classes = _make_class_info_step_tail(
        #     class_list=class_list,
        #     train_files=train_files,
        #     anomaly_files=anomaly_files,
        #     noise_ratio=noise_ratio,
        #     tail_k=tail_k,
        #     tail_class_ratio=tail_class_ratio,
        #     noise_on_tail=noise_on_tail,
        # )

    _save_data_config(files, train_files, anomaly_files, num_tail_samples, num_noise_samples, head_classes, data_config_path)

    _make_data(
        source_dir=source_dir,
        target_dir=target_dir,
        files=files,
        train_files=train_files,
        anomaly_files=anomaly_files,
        num_tail_samples=num_tail_samples,
        num_noise_samples=num_noise_samples,
        head_classes=head_classes,
    )


def make_data_pareto(
    source_dir: str,
    target_dir: str,
    noise_ratio: float = 0.1,
    noise_on_tail: bool = False,  # TODO: need to be implemented
    seed: int = 0,
) -> None:
    
    set_seed(seed)

    data_config_name = os.path.join(_DATA_CONFIG_ROOT, f'pareto_nr{int(noise_ratio*100):02d}_seed{seed}')

    if noise_on_tail:
        data_config_name += f"_tailnoised"
    
    data_config_path = f"{data_config_name}.pkl"

    # data_config_path = data_config_path.replace("pareto_", "mvtec_pareto_random_")
    
    if os.path.exists(data_config_path):
        files, train_files, anomaly_files, num_tail_samples, num_noise_samples, head_classes = _load_data_config(data_config_path)
    else:

        raise FileNotFoundError()

        _MVTEC_CLASS_LIST = get_subdirectories(source_dir)

        class_list = _MVTEC_CLASS_LIST
        files, train_files, anomaly_files = _get_mvtec_base_file_info(source_dir)
        num_tail_samples, num_noise_samples, head_classes = _make_class_info_pareto_tail(
            class_list, train_files, noise_ratio, noise_on_tail=noise_on_tail
        )
    
    _save_data_config(files, train_files, anomaly_files, num_tail_samples, num_noise_samples, head_classes, data_config_path)

    _make_data(
        source_dir=source_dir,
        target_dir=target_dir,
        files=files,
        train_files=train_files,
        anomaly_files=anomaly_files,
        num_tail_samples=num_tail_samples,
        num_noise_samples=num_noise_samples,
        head_classes=head_classes,
    )


def _make_data(
    source_dir,
    target_dir,
    files,
    train_files,
    anomaly_files,
    num_tail_samples,
    num_noise_samples,
    head_classes,
) -> None:

    tailed_files, noisy_files = _select_tailed_noises(
        files=files,
        train_files=train_files,
        anomaly_files=anomaly_files,
        num_tail_samples=num_tail_samples,
        num_noise_samples=num_noise_samples,
        head_classes=head_classes,
    )

    file_mapper_tail = _make_file_mapper(source_dir, target_dir, file_list=tailed_files)
    file_mapper_noise = _make_file_mapper(
        source_dir,
        target_dir,
        file_list=noisy_files,
        reflect_subfolder_depth=1,
        modify_subfolder_by={-2: "good", -3: "train"},
    )

    create_symlinks(file_mapper_tail)
    create_symlinks(file_mapper_noise)


def _get_mvtec_base_file_info(source_dir):
    # tail_files contain all file paths but with modified distributions on the tail classes
    files = {}
    train_files = {}
    test_files = {}
    anomaly_files = {}

    _MVTEC_CLASS_LIST = get_subdirectories(source_dir)

    for class_name in _MVTEC_CLASS_LIST:
        files[class_name] = list_files_in_folders(
            os.path.join(source_dir, class_name), source_dir, 
        )
        train_files[class_name] = list_files_in_folders(
            os.path.join(source_dir, class_name, "train"), source_dir, 
        )
        test_files[class_name] = list_files_in_folders(
            os.path.join(source_dir, class_name, "test"), source_dir, 
        )

        _test_good_files = list_files_in_folders(
            os.path.join(source_dir, class_name, "test", "good"), source_dir, 
        )
        anomaly_files[class_name] = [
            file for file in test_files[class_name] if file not in _test_good_files
        ]

    return files, train_files, anomaly_files


def _select_tailed_noises(
    files, train_files, anomaly_files, num_tail_samples, num_noise_samples, head_classes
):
    # select tailed samples
    tailed_files = {}
    for tail_class, num_samples in num_tail_samples.items():
        _remove_files = list_files_to_remove(train_files[tail_class], k=num_samples)
        tailed_files[tail_class] = [
            file for file in files[tail_class] if file not in _remove_files
        ]

    for head_class in head_classes:
        tailed_files[head_class] = files[head_class]

    # select noise samples to add
    noisy_files = {}
    for noisy_class, num_samples in num_noise_samples.items():
        noisy_files[noisy_class] = random.sample(
            anomaly_files[noisy_class], num_samples
        )

    tailed_files = [item for sublist in tailed_files.values() for item in sublist]
    noisy_files = [item for sublist in noisy_files.values() for item in sublist]

    return tailed_files, noisy_files


def _make_class_info_pareto_tail(
    class_list, train_files, noise_ratio, n_iter=100, noise_on_tail=True
):

    pareto_alpha = 6.0  # hard-coded
    target_class_dist = get_discrete_pareto_pmf(
        alpha=pareto_alpha, sampe_space_size=len(class_list)
    )
    num_train_samples = {}
    for train_class in train_files.keys():
        num_train_samples[train_class] = len(train_files[train_class])

    total_num_tail_samples = 0
    _target_class_dist = deepcopy(target_class_dist)
    for _ in range(n_iter):
        np.random.shuffle(_target_class_dist)
        _target_num_class_samples = redistribute_num_class_samples(
            list(num_train_samples.values()), _target_class_dist
        )
        if sum(_target_num_class_samples) > total_num_tail_samples:
            total_num_tail_samples = sum(_target_num_class_samples)
            target_num_class_samples = _target_num_class_samples

    num_tail_samples = {}
    for i, class_name in enumerate(train_files.keys()):
        num_tail_samples[class_name] = target_num_class_samples[i]

    min_size = 20
    if noise_on_tail:
        min_size = 1
    total_num_noise_samples = round(total_num_tail_samples * noise_ratio)
    num_noise_samples = sample_name2size(
        num_tail_samples, total_num_noise_samples, min_size
    )

    return num_tail_samples, num_noise_samples, []


# def sample_keys_from_dict_of_int(d, n_samples):
#     # Flatten the dictionary: [(key, value), ...]
#     flattened = [(key, "") for key, value in d.items() for _ in range(value)]

#     # Randomly sample n_samples elements from the flattened list
#     sampled = random.sample(flattened, n_samples)

#     # Extract and return keys and values of the sampled elements
#     num_samples_by_keys = defaultdict(int)

#     for key, _ in sampled:
#         num_samples_by_keys[key] += 1
#     return dict(num_samples_by_keys)

def sample_name2size(name2size, n_samples, min_size=20):
    # Flatten the dictionary: [(key, value), ...]
    flattened = [(key, value) for key, value in name2size.items() for _ in range(value)]

    random.shuffle(flattened)

    # Extract and return keys and values of the sampled elements
    num_samples_by_keys = defaultdict(int)

    counter = 0
    for class_name, class_size in flattened:
        if class_size >= min_size:
            num_samples_by_keys[class_name] += 1
            counter += 1
        
        if counter == n_samples:
            break

    return dict(num_samples_by_keys)


def redistribute_num_class_samples(
    num_class_samples: list, target_class_dist: list, min_num_samples: int = 5
) -> list:

    num_class_samples = np.array(num_class_samples)
    target_class_dist = np.array(target_class_dist)

    assert (num_class_samples > min_num_samples).all()

    idx_max = target_class_dist.argmax()
    factor = num_class_samples[idx_max] / target_class_dist[idx_max]
    desired_samples = np.round(target_class_dist * factor).astype(int)

    # Adjusting downwards only
    for i in range(len(desired_samples)):
        desired_samples[i] = max(
            min(num_class_samples[i], desired_samples[i]), min_num_samples
        )

    return desired_samples.tolist()


def get_discrete_pareto_pmf(alpha, sampe_space_size, epsilon=0.01):
    assert epsilon > 0
    x = np.linspace(
        pareto.ppf(epsilon, alpha), pareto.ppf(1 - epsilon, alpha), sampe_space_size + 1
    )
    cdf = pareto.cdf(x, alpha)
    pmf = (cdf - np.concatenate([[0], cdf])[:-1])[1:]

    return pmf


def _make_class_info_step_tail(
    class_list,
    train_files,
    anomaly_files,
    noise_ratio,
    tail_k,
    tail_class_ratio,
    noise_on_tail,
):
    assert not noise_on_tail
    _num_tail_classes = round(len(class_list) * tail_class_ratio)
    tail_classes = random.sample(class_list, _num_tail_classes)
    head_classes = [cls_name for cls_name in class_list if cls_name not in tail_classes]

    num_tail_samples = {}
    num_noise_samples = {}

    for tail_class in tail_classes:
        num_tail_samples[tail_class] = tail_k

    if not noise_on_tail:
        noisy_classes = head_classes

    for noisy_class in noisy_classes:
        num_noise_samples[noisy_class] = min(
            len(anomaly_files[noisy_class]),
            round(len(train_files[noisy_class]) * noise_ratio),
        )

    return num_tail_samples, num_noise_samples, head_classes


def _make_file_mapper(
    source_dir,
    target_dir,
    file_list=None,
    reflect_subfolder_depth=0,
    modify_subfolder_by=None,
):
    assert file_list is not None
    file_mapper = {}
    for file in file_list:
        _rel_path = file
        _source_file_path = os.path.join(source_dir, _rel_path)
        _target_file_path = os.path.join(target_dir, _rel_path)
        if reflect_subfolder_depth > 0:
            _target_file_path = os.path.join(
                os.path.dirname(_target_file_path),
                "_".join(_rel_path.split("/")[-(reflect_subfolder_depth + 1) :]),
            )

        if modify_subfolder_by:
            _target_file_path = modify_subfolders_in_path(
                _target_file_path, modify_subfolder_by
            )
        file_mapper[_source_file_path] = _target_file_path

    return file_mapper


def list_files_in_folders(directory, source_dir, exts=["png", "jpg", "jpeg", "JPG", "JPEG"]):
    """
    List all files with given extensions in a directory and its subdirectories.

    :param directory: The path to the directory to search in.
    :param source_dir: The source directory to calculate relative paths.
    :param exts: The file extensions to look for (default is ['png', 'jpg', 'jpeg']).
    :return: A list of paths to files with the specified extensions.
    """
    all_files = []
    for ext in exts:
        files = _list_files_in_folders(directory, source_dir, ext)
        all_files.extend(files)

    return all_files

def _list_files_in_folders(directory, source_dir, ext="png"):
    """
    List all files with a given extension in a directory and its subdirectories using glob.

    :param directory: The path to the directory to search in.
    :param ext: The file extension to look for (default is 'png').
    :return: A list of paths to files with the specified extension.
    """
    # Construct the search pattern
    search_pattern = os.path.join(directory, "**", f"*.{ext}")

    # Use glob to find files recursively
    files_with_ext = glob.glob(search_pattern, recursive=True)
    rel_files_with_ext = [os.path.relpath(path, source_dir) for path in files_with_ext]

    return rel_files_with_ext


def list_files_to_remove(file_list, k):
    """
    List file paths to be removed, keeping only K random files in the given directory.

    :param directory: The path to the directory.
    :param k: The number of files to keep.
    :return: A list of file paths to be removed.
    """
    # # List all files in the directory
    # all_files = [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
    all_files = deepcopy(file_list)

    # # If there are fewer or equal files than k, return an empty list as there's nothing to remove
    if len(all_files) <= k:
        return []

    # Randomly permute the list of files
    random.shuffle(all_files)

    # Select the first K files to keep
    files_to_keep = set(all_files[:k])

    # List the files not in the keep list
    files_to_remove = [file for file in all_files if file not in files_to_keep]

    return files_to_remove


def create_symlinks(file_mapper):
    """
    Create symbolic links based on the file_mapper dictionary.

    :param file_mapper: Dictionary where key is source path and value is target path.
    """
    for source, target in file_mapper.items():
        # Ensure the directory of the target path exists
        target_dir = os.path.dirname(target)
        os.makedirs(target_dir, exist_ok=True)

        os.symlink(source, target)

        # # Create a symlink from the source to the target
        # try:
        #     os.symlink(source, target)
        # except:
        #     pass
        # print(f"Symlink created: {source} -> {target}")


def compare_directories(
    original_dir, modified_dir, is_file_to_exclude, extension="*.png"
):
    # Function to check the relative depth of a file

    original_files = glob.glob(
        os.path.join(original_dir, "**", extension), recursive=True
    )
    original_files = [
        f for f in original_files if not is_file_to_exclude(f, original_dir)
    ]

    modified_files = glob.glob(
        os.path.join(modified_dir, "**", extension), recursive=True
    )
    modified_files = [
        f for f in modified_files if not is_file_to_exclude(f, modified_dir)
    ]

    # Check if the count of files is the same
    if len(original_files) != len(modified_files):
        return False, "Number of PNG files differ between directories."

    # Check each file
    for original_file in original_files:
        modified_file = original_file.replace(original_dir, modified_dir)

        # Check if the file exists in the modified directory
        if not modified_file in modified_files:
            return False, f"File {modified_file} does not exist in modified directory."

        # Check if file contents are the same
        if not os.path.samefile(original_file, modified_file):
            return False, f"File contents differ for {original_file}"

    return True, "Directories are equivalent except for 'train' folders."


def is_in_mvtec_train_folder(file_path, base_dir):
    rel_path = os.path.relpath(file_path, base_dir)
    parts = rel_path.split(os.sep)
    return "train" in parts and parts.index("train") == 1

def get_subdirectories(directory_path):
    """
    Returns a list of subdirectory names found in the given directory path.

    Args:
    directory_path (str): The path to the directory whose subdirectories are to be listed.

    Returns:
    list: A list of subdirectory names.
    """
    subdirectories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    return subdirectories



def main():
    # arguments
    # tail_type = "step"
    tail_type = "pareto"

    seed = 2

    tail_k = 4  # 4 or 1
    noise_on_tail = False
    noise_ratio = 0.1

    # mvtec
    # source_dir = "/home/jay/mnt/hdd01/data/image_datasets/anomaly_detection/mvtec_anomaly_detection"
    # target_dir = f"/home/jay/mnt/hdd01/data/image_datasets/anomaly_detection/symlink_mvtec_{tail_type}_nr{int(noise_ratio*100):02d}"

    # mvtec
    source_dir = "/home/jay/savespace/database/generic/mvtec"
    target_dir = f"/home/jay/savespace/database/SYMLINKS/project_tailedcore/mvtec/symlink_mvtec_{tail_type}_nr{int(noise_ratio*100):02d}"

    # # visa_
    # source_dir = "/home/jay/savespace/database/SYMLINKS/project_tailedcore/visa/full"
    # target_dir = f"/home/jay/savespace/database/SYMLINKS/project_tailedcore/visa/symlink_visa_{tail_type}_nr{int(noise_ratio*100):02d}"

    if tail_type == "step" and tail_k is not None:
        target_dir += f"_k{tail_k}"
    if noise_on_tail:
        target_dir += f"_tailnoised"

    target_dir += f"_seed{seed}"

    set_seed(seed)
    if tail_type == "step":
        make_data_step(
            source_dir,
            target_dir,
            tail_k=tail_k,
            noise_on_tail=noise_on_tail,
            seed=seed,
        )
    elif tail_type == "pareto":
        make_data_pareto(
            source_dir,
            target_dir,
            noise_on_tail=noise_on_tail,
            seed=seed,
        )

    compare_directories(
        source_dir, target_dir, is_file_to_exclude=is_in_mvtec_train_folder
    )


if __name__ == "__main__":
    
    main()
    
