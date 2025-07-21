"""
Sample detections with scores mainly around zero, for the purpose of agile modeling, 
while also including some more positive and negative values. 

Calculate an asymmetric Gaussian distribution shape based on the min and max values of the scores
Then sample a number of unique items from the scores that match this distribution shape.

"""
import numpy as np
import random
import matplotlib.pyplot as plt

def calculate_sigmas(min_val, max_val, neg_target_y=0.05, pos_target_y=0.05):
    """
    Calculates the standard deviations (sigmas) for an asymmetric Gaussian.

    The sigmas are calculated such that f(x) = neg_target_y at min_val and
    f(x) = pos_target_y at max_val.

    The formula is derived by solving f(x) = exp(-x² / 2σ²) for σ:
    σ = |x| / sqrt(-2 * ln(y))

    Args:
        min_val (float): The most negative value for calculation.
        max_val (float): The most positive value for calculation.
        neg_target_y (float): The desired probability/height at min_val.
        pos_target_y (float): The desired probability/height at max_val.

    Returns:
        tuple: A tuple containing (sigma_neg, sigma_pos).
    """
    if not (0 < neg_target_y < 1 and 0 < pos_target_y < 1):
        raise ValueError("Target y values must be between 0 and 1.")
    # if min_val > 0 or max_val < 0:
    #     raise ValueError("min_val must be negative and max_val must be positive.")

    min_val = min(min_val, 0)  # Ensure min_val is non-positive
    max_val = max(max_val, 0)  # Ensure max_val is non-negative

    # Calculate denominator for each side based on its target_y
    denom_neg = np.sqrt(-2 * np.log(neg_target_y))
    denom_pos = np.sqrt(-2 * np.log(pos_target_y))
    
    sigma_neg = abs(min_val) / denom_neg if min_val != 0 else 1.0
    sigma_pos = max_val / denom_pos if max_val != 0 else 1.0
    
    return sigma_neg, sigma_pos

def _internal_sampler(numbers, original_indices, k, min_val, max_val, neg_target_y, pos_target_y):
    """
    Internal helper function to perform the core sampling logic.
    Selects k unique items from a list by matching a target distribution shape
    and returns their original indices. This version uses a more efficient
    direct sampling method instead of rejection sampling.
    """
    if k > len(numbers):
        raise ValueError(f"Sample size k ({k}) cannot be larger than the number of available items ({len(numbers)}).")
        
    # 1. Define the target probability distribution function (PDF)
    sigma_neg, sigma_pos = calculate_sigmas(min_val, max_val, neg_target_y, pos_target_y)
    print(f"Calculated σ_neg: {sigma_neg:.4f} (for x={min_val}, y={neg_target_y})")
    print(f"Calculated σ_pos: {sigma_pos:.4f} (for x={max_val}, y={pos_target_y})\n")

    def asymmetric_gaussian_pdf(x):
        sigma = np.where(x < 0, sigma_neg, sigma_pos)
        return np.exp(-(x**2) / (2 * sigma**2))

    # Create a discrete approximation of the target PDF
    # num must be many times larger than k to ensure that when sampling without replacement
    # the sampled points resemble the distrubution shape.
    x_range = np.linspace(min_val, max_val, num=k*20) 
    pdf_weights = asymmetric_gaussian_pdf(x_range)
    # Normalize weights to use in np.random.choice
    pdf_weights /= np.sum(pdf_weights)

    # Draw all k ideal samples
    # We set replace=False to encourage a wider spread of ideal points to map from.
    ideal_samples = np.random.choice(x_range, size=k, replace=False, p=pdf_weights)

    # 3. MAP IDEAL SAMPLES TO REAL, UNIQUE DATA POINTS
    selected_original_indices = []
    available_items = list(zip(numbers, original_indices))
    
    print(f"Starting to map {k} ideal samples to available data points...")
    for ideal_val in ideal_samples:
        if not available_items:
            print("Warning: Ran out of available items to sample from.")
            break
        
        # Find the closest available real number to the current ideal sample
        available_values = np.array([item[0] for item in available_items])
        diffs = np.abs(available_values - ideal_val)
        closest_idx_in_available = np.argmin(diffs)
        
        # Get the chosen item (value, original_index) and remove it from the pool
        chosen_item = available_items.pop(closest_idx_in_available)
        
        # Add its original index to our results
        selected_original_indices.append(chosen_item[1])
            
    print(f"Successfully selected {len(selected_original_indices)} unique items.")
    return np.array(selected_original_indices)

def sample_for_validation(scores,
                          pos_threshold=None,
                          pos_target_y=0.05,
                          neg_target_y=0.05,
                          neg_threshold=None,
                          sample_size=100):
    """
    Filters a list of scores and selects a sample whose distribution matches
    a target asymmetric Gaussian shape. Returns the original indices of the samples.

    Args:
        scores (list or np.ndarray): The input list of scores.
        pos_threshold (float, optional): Exclude scores above this value.
        pos_target_y (float, optional): Target probability at the positive end.
        neg_target_y (float, optional): Target probability at the negative end.
        neg_threshold (float, optional): Exclude scores below this value.
        sample_size (int, optional): The number of unique samples to select.

    Returns:
        tuple: (sampled_indices, filtered_scores, effective_min, effective_max)
               Contains the original indices of the final sample, the data after filtering,
               and the min/max values used for sigma calculation.
    """
    scores_arr = np.array(scores)
    original_indices = np.arange(len(scores_arr))
    
    # 1. Create a boolean mask for filtering scores and their indices together
    mask = np.ones_like(scores_arr, dtype=bool)
    if neg_threshold is not None:
        mask &= (scores_arr >= neg_threshold)
    if pos_threshold is not None:
        mask &= (scores_arr <= pos_threshold)

    filtered_scores = scores_arr[mask]
    filtered_original_indices = original_indices[mask]

    if len(filtered_scores) == 0:
        raise ValueError("No scores remain after applying thresholds.")

    # 2. Determine the effective min/max for the distribution shape
    # Use the threshold if provided, otherwise use the data's actual min/max
    effective_min = neg_threshold if neg_threshold is not None else filtered_scores.min()
    effective_max = pos_threshold if pos_threshold is not None else filtered_scores.max()

    # if we don't have enough to sample, adjust sample_size
    # this will select them all, but it's easier to just do it than have a special condition
    sample_size = min(sample_size, len(filtered_scores))

    # 3. Call the internal sampler with the filtered data and their original indices
    sampled_indices = _internal_sampler(
        numbers=filtered_scores,
        original_indices=filtered_original_indices,
        k=sample_size,
        min_val=effective_min,
        max_val=effective_max,
        neg_target_y=neg_target_y,
        pos_target_y=pos_target_y
    )
    
    return sampled_indices, filtered_scores, effective_min, effective_max

def plot_sampling_results(original_scores, sampled_scores, filtered_scores, min_val, max_val, neg_target_y, pos_target_y):
    """
    Generates a 3-panel plot to visualize the sampling process and results.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig.suptitle('Sampling to Match a Target Distribution Shape', fontsize=16)
    
    # Determine common bins for all histograms
    bins = np.linspace(original_scores.min(), original_scores.max(), 100)

    # Plot 1: The original, non-uniform data distribution
    ax1.hist(original_scores, bins=bins, color='lightgray', label='Original Data')
    ax1.hist(filtered_scores, bins=bins, color='gray', label='Filtered Data (Used for Sampling)')
    ax1.set_title('1. Original and Filtered Data Distribution')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: The theoretical probability distribution we are aiming for
    sigma_neg, sigma_pos = calculate_sigmas(min_val, max_val, neg_target_y, pos_target_y)
    x_vals = np.linspace(min_val, max_val, 500)
    target_weights = [np.exp(-(x**2) / (2 * (sigma_neg if x < 0 else sigma_pos)**2)) for x in x_vals]
    ax2.plot(x_vals, target_weights, 'b-', label='Target Shape f(x)')
    ax2.set_title('2. Target Asymmetric Gaussian Distribution Shape')
    ax2.set_ylabel('Target Probability')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    # Plot 3: A histogram of the actual sampled values
    ax3.hist(sampled_scores, bins=bins, color='skyblue', label='Sampled Values')
    ax3.set_title(f'3. Final Sampled Distribution (k={len(sampled_scores)})')
    ax3.set_xlabel('Number')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def demonstrate_sampling():
    """
    Creates dummy data and runs the full sampling and plotting process.
    """
    # 1. DEFINE YOUR DATA
    # Create a NON-UNIFORM list of numbers to demonstrate the method's power.
    group1 = np.random.normal(loc=-40, scale=10, size=300)
    group2 = np.random.normal(loc=60, scale=20, size=700)
    my_scores = np.concatenate([group1, group2]).astype(int)
    
    # 2. PERFORM SAMPLING
    sampled_indices, filtered_data, eff_min, eff_max = sample_for_validation(
        scores=my_scores,
        neg_threshold=-60,      # Ignore scores below -60
        pos_threshold=110,      # Ignore scores above 110
        neg_target_y=0.1,       # Aim for a height of 0.1 at x=-60
        pos_target_y=0.05,      # Aim for a height of 0.05 at x=110
        sample_size=150
    )
    
    # Get the actual values of the sampled scores using the returned indices
    sampled_values = my_scores[sampled_indices]
    
    print(f"\nSample of 10 chosen indices: {sampled_indices[:10]}")
    print(f"Corresponding values: {sampled_values[:10]}")

    # 3. VISUALIZE THE RESULTS
    plot_sampling_results(
        original_scores=my_scores,
        sampled_scores=sampled_values,
        filtered_scores=filtered_data,
        min_val=eff_min,
        max_val=eff_max,
        neg_target_y=0.1,
        pos_target_y=0.05
    )

if __name__ == "__main__":
    demonstrate_sampling()
