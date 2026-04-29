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
