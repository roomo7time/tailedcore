def test_greedy_coreset_sampler():
    import torch
    import numpy as np
    from src.sampler import GreedyCoresetSampler

    percentage = 0.01
    dimension_to_project_features_to = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    brute = False
    max_coreset_len = 1000  # Example parameter

    # Initialize the sampler
    sampler = GreedyCoresetSampler(
        percentage, dimension_to_project_features_to, device, brute, max_coreset_len
    )

    # Generate sample data (features) using numpy
    num_samples = 100  # Example number of samples
    feature_dim = 20  # Example feature dimension
    features = np.random.randn(num_samples, feature_dim)

    # Optionally, create a base coreset using numpy
    base_coreset_size = 2000  # Example base coreset size
    base_coreset = np.random.randn(base_coreset_size, feature_dim)

    # Run the sampler
    sampled_features, sample_indices = sampler.run(features, base_coreset)

    # Print results
    print("Sampled Features Shape:", sampled_features.shape)
    print("Sample Indices:", sample_indices)
