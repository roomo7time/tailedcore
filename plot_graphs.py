import torch


def get_extracted_artifacts():

    extracted = _get_extracted_artifacts_mvtec_step_nr10_tk4_seed0_wrn50()

    # feas = extracted["feas"]
    # masks = extracted["masks"]

    # gaps = extracted["gaps"]
    # class_names = extracted["class_names"]
    # class_sizes = extracted["class_sizes"]

    return extracted


def _get_extracted_artifacts_mvtec_step_nr10_tk4_seed0_wrn50():

    extracted_path = (
        "./shared_resources/extracted_mvtec_step_nr10_tk4_tr60_seed0_wrn50.pt"
    )

    extracted = torch.load(extracted_path)

    return extracted


def plot_class_size_vs_density():

    extracted = get_extracted_artifacts()

    feas = extracted["feas"]
    masks = extracted["masks"]
    rmasks = extracted["downsized_masks"]

    gaps = extracted["gaps"]
    class_names = extracted["class_names"]
    class_sizes = extracted["class_sizes"]

    b, fea_dim, h, w = feas.size()
    is_anomaly_gt = torch.round(rmasks).reshape((-1, ))
    patch_class_sizes = class_sizes[:, None].repeat(1, h*w).reshape((-1, ))
    patch_features = feas.reshape(b, fea_dim, -1).permute(0, 2, 1).reshape((-1, fea_dim))




    


if __name__ == "__main__":

    plot_class_size_vs_density()
