import numpy as np 
import matplotlib.pyplot as plt
import math as m
from scipy.stats import norm, poisson, expon, kstest
import statistics as st

def detect_moderate_outliers(data_list: list):
    """
    Identifies moderate outliers using the 1.5 * IQR rule.
    Returns a list of values that fall outside the calculated bounds.
    """
    # Calculate quartiles and Interquartile Range (IQR)
    q1 = np.percentile(data_list, 25)
    q3 = np.percentile(data_list, 75)
    iqr = q3 - q1
    
    # Define bounds for moderate outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = [value for value in data_list if value < lower_bound or value > upper_bound]
    return outliers

def remove_moderate_outliers(data_list: list):
    """
    Returns a list excluding values outside the 1.5 * IQR range.
    """
    q1 = np.percentile(data_list, 25)
    q3 = np.percentile(data_list, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    clean_list = [value for value in data_list if lower_bound <= value <= upper_bound]
    return clean_list

def detect_extreme_outliers(data_list: list):
    """
    Identifies extreme outliers using the 3 * IQR rule.
    """
    q1 = np.percentile(data_list, 25)
    q3 = np.percentile(data_list, 75)
    iqr = q3 - q1
    
    # Bounds for extreme cases (3x multiplier)
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    
    outliers = [value for value in data_list if value < lower_bound or value > upper_bound]
    return outliers

def remove_extreme_outliers(data_list: list):
    """
    Removes extreme outliers (3 * IQR) and prints statistical thresholds.
    """
    q1 = np.percentile(data_list, 25)
    q3 = np.percentile(data_list, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    
    # Logging the thresholds for debugging
    print(f"Stats Log - Q1: {q1}, Q3: {q3}, IQR: {iqr}")
    print(f"Bounds - Lower: {lower_bound}, Upper: {upper_bound}")
    
    clean_list = [value for value in data_list if lower_bound <= value <= upper_bound]
    return clean_list

def get_frequency(data_list: list) -> dict: 
    """
    Calculates absolute frequency (count) for each unique item.
    """
    data_list.sort()
    freq = {} 
    for item in data_list: 
        freq[item] = freq.get(item, 0) + 1
    return freq

def get_cumulative_frequency(freq_dict: dict) -> dict:
    """
    Calculates cumulative frequency by summing up counts in sorted order.
    """
    cumulative_dict = {}
    running_total = 0
    # Iterating through sorted keys to ensure correct summation
    for key in sorted(freq_dict.keys()):
        running_total += freq_dict[key]
        cumulative_dict[key] = running_total
    return cumulative_dict

def normal_pdf(x, mu, sigma):
    """
    Mathematical implementation of the Normal Probability Density Function.
    Formula: 1 / (σ * sqrt(2π)) * e^(-0.5 * ((x-μ)/σ)^2)
    """
    coefficient = 1 / (sigma * m.sqrt(2 * m.pi))
    exponent = -((x - mu) ** 2) / (2 * (sigma ** 2))
    return coefficient * m.exp(exponent)

def get_pdfs(values: list) -> list: 
    """
    Generates a list of PDF values for each item based on the list's mean and std dev.
    """
    std_dev = st.pstdev(values)
    mean_val = st.mean(values)
    return [normal_pdf(i, mean_val, std_dev) for i in values]

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
filtered_data = remove_extreme_outliers(sorted_data)
abs_frequency = get_frequency(filtered_data)
cum_frequency = get_cumulative_frequency(abs_frequency)

# Statistical Analysis
data_mean = np.mean(filtered_data)
data_std = np.std(filtered_data)

# Kolmogorov-Smirnov Test Setup
distributions = {
    "Normal": ("norm", (data_mean, data_std)),
    "Exponential": ("expon", (0, data_mean)),
    "Poisson": ("poisson", (data_mean,))
}

print("\n--- GOODNESS-OF-FIT TEST RESULTS ---")
for name, (dist_code, params) in distributions.items():
    # Perform Kolmogorov-Smirnov test
    stat, p_val = kstest(filtered_data, dist_code, args=params)
    
    # Decision rule: if p-value > 0.05, we fail to reject H0 (Good fit)
    status = "ACCEPTED" if p_val > 0.005 else "REJECTED"
    print(f"{name:<12}: P-value = {p_val:.5f} ({status} H0)")