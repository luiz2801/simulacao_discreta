import numpy as np 
import matplotlib.pyplot as plt
import math as m
from scipy.stats import norm
import statistics as st

def detect_moderate_outliers(data_list: list):
    """
    To calculate a moderate outlier, you need to find the first and third quartiles.
    The difference between them (IQR) is multiplied by 1.5.
    Any value with a difference greater than this range is a moderate outlier.
    """
    # First quartile
    q1 = np.percentile(data_list, 25)
    # Third quartile
    q3 = np.percentile(data_list, 75)
    # Interquartile range
    iqr = q3 - q1
    # Outlier boundaries
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = []
    for value in data_list:
        if value < lower_bound or value > upper_bound:
            outliers.append(value)

    return outliers

def remove_moderate_outliers(data_list: list):
    """
    Calculates the moderate outlier boundaries and returns a list 
    excluding any values outside the 1.5 * IQR range.
    """
    q1 = np.percentile(data_list, 25)
    q3 = np.percentile(data_list, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    clean_list = []
    for value in data_list:
        if lower_bound <= value <= upper_bound:
            clean_list.append(value)

    return clean_list

def detect_extreme_outliers(data_list: list):
    """
    To calculate an extreme outlier, the IQR is multiplied by 3.
    Any value beyond this threshold is considered an extreme outlier.
    """
    q1 = np.percentile(data_list, 25)
    q3 = np.percentile(data_list, 75)
    iqr = q3 - q1
    
    # Outlier boundaries (3 times the IQR for extremes)
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    
    outliers = []
    for value in data_list:
        if value < lower_bound or value > upper_bound:
            outliers.append(value)

    return outliers

def remove_extreme_outliers(data_list: list):
    """
    Calculates the extreme outlier boundaries and returns a list 
    excluding any values outside the 3 * IQR range.
    """
    q1 = np.percentile(data_list, 25)
    q3 = np.percentile(data_list, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    
    print(f"Upper bound is {upper_bound}")
    print(f"Lower bound is {lower_bound}")
    print(f"Q1 is {q1}")
    print(f"Q3 is {q3}")    
    
    clean_list = []
    for value in data_list:
        if lower_bound <= value <= upper_bound:
            clean_list.append(value)

    return clean_list

def get_frequency(data_list: list) -> dict: 
    """
    Calculates the absolute frequency of each element in the list.
    Returns a dictionary where the key is the element and the value is the count.
    """
    data_list.sort()
    freq = {} 
    
    for item in data_list: 
        if item in freq: 
            freq[item] += 1
        else: 
            freq[item] = 1

    return freq

def get_cumulative_frequency(freq_dict: dict) -> dict:
    """
    Calculates the cumulative frequency from an absolute frequency dictionary.
    Iterates through sorted keys to perform a progressive summation.
    """
    cumulative_dict = {}
    running_total = 0
    # Sort keys to ensure accumulation follows numerical/alphabetical order
    for key in sorted(freq_dict.keys()):
            running_total += freq_dict[key]
            cumulative_dict[key] = running_total
    return cumulative_dict

def normal_pdf(x, mu, sigma):
    """
    Calculates the Probability Density Function (PDF) of the Normal Distribution.
    Uses the mathematical formula: 
    1 / (sigma * sqrt(2 * pi)) * exp(-0.5 * ((x - mu) / sigma)^2)
    """
    coefficient = 1 / (sigma * m.sqrt(2 * m.pi))
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    
    return coefficient * m.exp(exponent)

lista = [
    11, 5, 2, 0, 9, 9, 1, 5, 1, 3,
    3, 3, 7, 4, 12, 8, 5, 2, 6, 1,
    11, 1, 2, 4, 2, 1, 3, 9, 0, 10,
    3, 3, 1, 5, 18, 4, 22, 8, 3, 0,
    8, 9, 2, 3, 12, 1, 3, 1, 7, 5,
    14, 7, 7, 28, 1, 3, 2, 11, 13, 2,
    0, 1, 6, 12, 15, 0, 6, 7, 19, 1,
    1, 9, 1, 5, 3, 17, 10, 15, 43, 2,
    6, 1, 13, 13, 19, 10, 9, 20, 19, 2,
    27, 5, 20, 5, 10, 8, 2, 3, 1, 1,
    4, 3, 6, 13, 10, 9, 1, 1, 3, 9,
    9, 4, 0, 3, 6, 3, 27, 3, 18, 4,
    6, 0, 2, 2, 8, 4, 5, 1, 4, 18,
    1, 0, 16, 20, 2, 2, 2, 12, 28, 0,
    7, 3, 18, 12, 3, 2, 8, 3, 19, 12,
    5, 4, 6, 0, 5, 0, 3, 7, 0, 8,
    8, 12, 3, 7, 1, 3, 1, 3, 2, 5,
    4, 9, 4, 12, 4, 11, 9, 2, 0, 5,
    8, 24, 1, 5, 12, 9, 17, 728, 12, 6,
    4, 3, 5, 7, 4, 4, 4, 11, 3, 8
]

organizada = sorted(lista)
sem_outliers = remove_outlier_extremo(organizada)
frequencia = frequency(sem_outliers)
freq_acumulada = sum_freq(frequencia)
# distribuicao = np.random.normal(scale = freq_acumulada)
sem_rep = set(sem_outliers)

def get_pdfs(valores: list) -> list: 
    # Desvio padrão 
    dp = st.pstdev(valores)
    lista = []
    for i in valores:
        lista.append(normal_pdf(i, st.mean(valores), dp))
    return lista    
# print("olha aqui", get_pdfs(sem_outliers))




