def test_get_max_len_feas():
    from src import automl
    import numpy as np

    # Test with different feature dimensions and devices
    fea_dim = 2048
    max_len = automl.get_max_len_feas(fea_dim, max_usage=0.5)

    dummy_mat = np.random.rand(max_len, fea_dim)
    memory_usage = automl.calculate_memory_usage(dummy_mat)
    print(f"memory_usage: {memory_usage}")
    # for fea_dim in [100, 1000, 10000]:
    #     for device in ['cpu', 'cuda']:
    #         max_len = get_max_len_feas(fea_dim, device)
    #         print(f"Max length of features for fea_dim {fea_dim} on {device}: {max_len}")


def test_get_max_coreset_ratio():
    from src.automl import get_max_coreset_ratio

    # Test with different feature dimensions and lengths
    for fea_dim in [100, 1000]:
        for len_feas in [10000, 20000, 50000]:
            max_coreset_ratio = get_max_coreset_ratio(fea_dim, len_feas)
            print(
                f"Max coreset ratio for fea_dim {fea_dim} and len_feas {len_feas}: {max_coreset_ratio}"
            )
