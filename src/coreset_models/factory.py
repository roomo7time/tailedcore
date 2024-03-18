
from .base import BaseCore
from .patchcore import PatchCore
from .softpatch import SoftPatch
from .tailedpatch import TailedPatch
from .constantcore import ConstantCore

def get_coreset_model(
    model_config,
    feature_embedder,
    imagesize,
    device,
    faiss_on_gpu,
    faiss_num_workers,
    sampler_on_gpu,
    save_dir_path=None,
    brute=False,
    coreset_ratio=0.01,
    max_coreset_size=None,
    **kwargs,
) -> BaseCore:

    coreset_ratio = getattr(model_config, "greedy_ratio", coreset_ratio)
    if model_config.coreset_model_name == "constantcore":
        return ConstantCore(
            feature_embedder,
            device,
            imagesize,
            coreset_ratio=coreset_ratio,
            faiss_on_gpu=faiss_on_gpu,
            faiss_num_workers=faiss_num_workers,
            sampler_on_gpu=sampler_on_gpu,
            save_dir_path=save_dir_path,
            brute=brute,
            max_coreset_size=max_coreset_size,
        )
    elif model_config.coreset_model_name == "softpatch":
        return SoftPatch(
            feature_embedder,
            device,
            imagesize,
            coreset_ratio=coreset_ratio,
            faiss_on_gpu=faiss_on_gpu,
            faiss_num_workers=faiss_num_workers,
            sampler_on_gpu=sampler_on_gpu,
            save_dir_path=save_dir_path,
            brute=brute,
        )
    elif model_config.coreset_model_name == "patchcore":
        return PatchCore(
            feature_embedder,
            device,
            imagesize,
            coreset_ratio=coreset_ratio,
            faiss_on_gpu=faiss_on_gpu,
            faiss_num_workers=faiss_num_workers,
            sampler_on_gpu=sampler_on_gpu,
            save_dir_path=save_dir_path,
            brute=brute,
        )
    elif model_config.coreset_model_name in ["tailedpatch", "aatailedpatch"]:
        return TailedPatch(
            feature_embedder,
            device,
            imagesize,
            coreset_ratio=coreset_ratio,
            faiss_on_gpu=faiss_on_gpu,
            faiss_num_workers=faiss_num_workers,
            sampler_on_gpu=sampler_on_gpu,
            save_dir_path=save_dir_path,
            brute=brute,
            tail_th_type=model_config.tail_th_type,
            tail_data_augment_type=model_config.tail_data_augment_type,
            tail_lof=model_config.tail_lof,
        )
    else:
        raise NotImplementedError()