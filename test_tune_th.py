import numpy as np
from src.evaluator.th_tuner import tune_score_threshold_to_minimize_fnfp

if __name__ == "__main__":
    # Create a sample batch of score masks and ground truth masks
    np.random.seed(0)
    batch_size = 5
    score_masks = np.random.rand(
        batch_size, 10, 10
    )  # Batch of random values between 0 and 1
    masks_gt = np.random.randint(
        2, size=(batch_size, 10, 10)
    )  # Batch of random binary values

    # Test the function
    best_threshold = tune_score_threshold_standard(score_masks, masks_gt, 1)
    print("Best threshold for the batch:", best_threshold)
