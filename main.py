import manipulate as mani
import distribuicoes as dist
import numpy as np
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

# Data Processing
sorted_data = sorted(raw_data)
filtered_data = mani.remove_extreme_outliers(sorted_data)
abs_frequency = mani.get_frequency(filtered_data)
cum_frequency = mani.get_cumulative_frequency(abs_frequency)


# Normalização (Standardization)
data_normalized = (filtered_data - np.mean(filtered_data)) / np.std(filtered_data)


# Statistical Analysis
data_mean = np.mean(data_normalized)
data_std = np.std(data_normalized)




# Kolmogorov-Smirnov Test Setup
distributions = {
    "Normal": ("norm", (data_mean, data_std)),
    "Exponential": ("expon", (0, data_mean)),
    "Poisson": ("poisson", (data_mean,))
}




print("\n--- GOODNESS-OF-FIT TEST RESULTS ---")
for name, (dist_code, params) in distributions.items():
    # Perform Kolmogorov-Smirnov test
    stat, p_val = kstest(data_normalized, dist_code, args=params)
    
    # Decision rule: if p-value > 0.005, we fail to reject H0 (Good fit)
    status = "ACCEPTED" if p_val > 0.005 else "REJECTED"
    print(f"{name:<12}: P-value = {p_val:.5f} ({status} H0)")