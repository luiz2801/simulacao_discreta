import manipulate as mani  # Assuming these modules are also renamed or updated
import distribuicoes as dist
import numpy as np
from scipy import stats
from scipy.stats import norm, poisson, expon, kstest

# --- Main Execution ---

raw_data = [
    11, 5, 2, 0, 9, 9, 1, 5, 1, 3, 3, 3, 7, 4, 12, 8, 5, 2, 6, 1,
    11, 1, 2, 4, 2, 1, 3, 9, 0, 10, 3, 3, 1, 5, 18, 4, 22, 8, 3, 0,
    8, 9, 2, 3, 12, 1, 3, 1, 7, 5, 14, 7, 7, 28, 1, 3, 2, 11, 13, 2,
    0, 1, 6, 12, 15, 0, 6, 7, 19, 1, 1, 9, 1, 5, 3, 17, 10, 15, 43, 2,
    6, 1, 13, 13, 19, 10, 9, 20, 19, 2, 27, 5, 20, 5, 10, 8, 2, 3, 1, 1,
    4, 3, 6, 13, 10, 9, 1, 1, 3, 9, 9, 4, 0, 3, 6, 3, 27, 3, 18, 4,
    6, 0, 2, 2, 8, 4, 5, 1, 4, 18, 1, 0, 16, 20, 2, 2, 2, 12, 28, 0,
    7, 3, 18, 12, 3, 2, 8, 3, 19, 12, 5, 4, 6, 0, 5, 0, 3, 7, 0, 8,
    8, 12, 3, 7, 1, 3, 1, 3, 2, 5, 4, 9, 4, 12, 4, 11, 9, 2, 0, 5,
    8, 24, 1, 5, 12, 9, 17, 728, 12, 6, 4, 3, 5, 7, 4, 4, 4, 11, 3, 8
]

 
raw_data.sort()

# --- Data Processing ---
sorted_data = sorted(raw_data)
filtered_data = mani.remove_extreme_outliers(sorted_data)
abs_frequency = mani.get_frequency(filtered_data)
cum_frequency = mani.get_cumulative_frequency(abs_frequency)

# --- Standardization (Normalization) ---
data_normalized = (filtered_data - np.mean(filtered_data)) / np.std(filtered_data)

# --- Statistical Analysis ---
data_mean = np.mean(data_normalized)
data_std = np.std(data_normalized)

# --- Goodness of Fit Test ---
distributions = ("uniform", "expo", "lognorm", "triang", "norm")

# Mapping to ensure distribution names match scipy.stats methods
dist_mapping = {
    "uniform": stats.uniform,
    "expo": stats.expon,
    "lognorm": stats.lognorm,
    "triang": stats.triang,
    "norm": stats.norm
}

print(f"{'Distribution':<12} | {'P-Value':<10} | {'KS Statistic':<15} | {'Result'}")
print("-" * 65)

for item in distributions:
    # 1. Get the scipy distribution object
    dist_object = dist_mapping[item]
    
    # 2. Fit the distribution to the data
    # This estimates the best parameters (shape, location, scale)
    params = dist_object.fit(data_normalized)
    
    # 3. Kolmogorov-Smirnov Test
    # Compares empirical data against the theoretical distribution
    dist_name = item if item != "expo" else "expon"
    ks_stat, p_value = stats.kstest(data_normalized, dist_name, args=params)
    
    # 4. Interpretation
    # Null Hypothesis (H0): Data follows the distribution. 
    # If p_value > 0.05, we fail to reject H0.
    fit_quality = "Good Fit" if p_value > 0.05 else "Poor Fit"
    
    print(f"{item:<12} | {p_value:.4e} | {ks_stat:.4f}          | {fit_quality}") 