import math as m
import numpy as np

# ==========================================
# 1. UNIFORM DISTRIBUTION
# Formula: 1 / (b - a)
# ==========================================

def uniform_pdf(x: float, a: float, b: float) -> float:
    """Calculates the Uniform probability density for a value x."""
    if a <= x <= b:
        return 1 / (b - a)
    return 0.0

def generate_uniform_data(size: int, a: float, b: float) -> np.ndarray:
    """Generates a dataset following the Uniform distribution."""
    return np.random.uniform(a, b, size)


# ==========================================
# 2. EXPONENTIAL DISTRIBUTION
# Formula: lambda * exp(-lambda * x)
# ==========================================

def exponential_pdf(x: float, lambd: float) -> float:
    """Calculates the Exponential probability density for a value x."""
    if x >= 0:
        return lambd * m.exp(-lambd * x)
    return 0.0

def generate_exponential_data(size: int, lambd: float) -> np.ndarray:
    """
    Generates data in the Exponential distribution.
    Note: In numpy, the 'scale' parameter is the inverse of lambda (1 / lambda).
    """
    return np.random.exponential(scale=(1/lambd), size=size)


# ==========================================
# 3. LOGNORMAL DISTRIBUTION
# Formula: (1 / (x * sigma * sqrt(2*pi))) * exp(-((ln(x) - mu)^2) / (2*variance))
# ==========================================

def lognormal_pdf(x: float, mu: float, variance: float) -> float:
    """Calculates the Lognormal probability density for a value x > 0."""
    if x <= 0:
        return 0.0
    sigma = m.sqrt(variance)
    coefficient = 1 / (x * sigma * m.sqrt(2 * m.pi))
    exponent = -((m.log(x) - mu) ** 2) / (2 * variance)
    return coefficient * m.exp(exponent)

def generate_lognormal_data(size: int, mu: float, variance: float) -> np.ndarray:
    """Generates data following the Lognormal distribution."""
    sigma = m.sqrt(variance)
    return np.random.lognormal(mean=mu, sigma=sigma, size=size)


# ==========================================
# 4. TRIANGULAR DISTRIBUTION
# Formula: 
# If a <= x < mode: 2 * (x - a) / ((b - a) * (mode - a))
# If mode <= x <= b: 2 * (b - x) / ((b - a) * (b - mode))
# ==========================================

def triangular_pdf(x: float, a: float, mode: float, b: float) -> float:
    """Calculates the Triangular probability density for a value x."""
    if x < a or x > b:
        return 0.0
    elif a <= x < mode:
        return 2 * (x - a) / ((b - a) * (mode - a))
    elif x == mode:
        return 2 / (b - a)
    else:  # mode < x <= b
        return 2 * (b - x) / ((b - a) * (b - mode))

def generate_triangular_data(size: int, a: float, mode: float, b: float) -> np.ndarray:
    """Generates data forming a Triangular distribution."""
    return np.random.triangular(left=a, mode=mode, right=b, size=size)

# ==========================================
# 5. NORMAL DISTRIBUTION
# ==========================================

def normal_pdf(x: float, mu: float, sigma: float) -> float:
    """
    Calculates the Normal probability density for a value x.
    Formula: 1 / (sigma * sqrt(2*pi)) * exp(-0.5 * ((x-mu)/sigma)^2)
    """
    if sigma <= 0:
        raise ValueError("Standard deviation (sigma) must be greater than zero.")
    
    coefficient = 1 / (sigma * m.sqrt(2 * m.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coefficient * m.exp(exponent)

def generate_normal_data(size: int, mu: float, sigma: float) -> np.ndarray:
    """
    Generates a data array following the Normal distribution.
    """
    return np.random.normal(loc=mu, scale=sigma, size=size)