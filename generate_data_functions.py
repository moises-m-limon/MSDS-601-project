import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_point, geom_smooth, labs, geom_line
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
import plotly.graph_objs as go


def calculate_summary_statistics(data, x_column, y_column):
    summary_stats = {
        "Mean": [data[x_column].mean(), data[y_column].mean()],
        "Median": [data[x_column].median(), data[y_column].median()],
        "Standard Deviation": [data[x_column].std(ddof=1), data[y_column].std(ddof=1)],
        "Min": [data[x_column].min(), data[y_column].min()],
        "Max": [data[x_column].max(), data[y_column].max()],
        "Count": [len(data[x_column]), len(data[y_column])],
        "Correlation": [data[x_column].corr(data[y_column])],
    }
    return pd.DataFrame(summary_stats, index=["X", "Y"]).T


# Function to generate multicollinear data
def generate_multicollinear_data(n_samples, noise, z_noise, noise_type):
    np.random.seed(42)
    x = np.random.uniform(1, 10, n_samples)

    if noise_type == "Heteroskedastic":
        noise = noise * x/2

    y = x + np.random.normal(0.0, noise, n_samples)
    
    z = (x + y)/2 + np.random.normal(0.0, z_noise, n_samples)
    
    # Return a DataFrame with the generated data
    return pd.DataFrame({"x": x, "y": y, "z": z})

# Function to generate non-multicollinear data
def generate_non_multicollinear_data(z_noise, n=10):
    np.random.seed(42)
    x = np.random.uniform(1, 10, n)
    y = np.random.uniform(1, 10, n)  # Independent of x1
    z =  (x + y)/2 + np.random.normal(0.0, z_noise, n)

    return pd.DataFrame({"x": x, "y": y, "z": z})


def generate_data_v2(y_expression="x**2", n=100, xmin=-20, xmax=50, sigma=700):
    np.random.seed(42)
    x = np.linspace(xmin, xmax, n)

    # Evaluate y based on the expression
    y = eval(y_expression) + np.random.normal(0, sigma, n)

    return pd.DataFrame({"x": x, "y": y})

# Function to generate synthetic data
def generate_synthetic_data(n_samples, intercept, slope, x_min, x_max, error_dist, noise_level, noise_type):

    np.random.seed(42)  # for reproducability in a classroom?

    x = np.random.uniform(low=x_min, high=x_max, size=n_samples)
    # use distr from selectbox
    if error_dist == "Normal":
        noise = np.random.normal(0, noise_level, n_samples)
    elif error_dist == "Uniform":
        noise = np.random.uniform(-noise_level, noise_level, n_samples)
    elif error_dist == "t":
        noise = np.random.standard_t(df=10, size=n_samples) * noise_level
    elif error_dist == 'random':
        noise = (np.random.random(n_samples) - 0.5) * 2 * noise_level
    if noise_type == "Heteroskedastic":
        noise = noise * x/2

    y = intercept + slope * x + noise
    operations = ['x**2', 'np.log(x)', 'np.sqrt(x)', 'np.exp(x)']


    func = np.random.choice(operations)
    y2 = intercept + slope * eval(func) + noise

    return pd.DataFrame({"x": x, "y": y, "qx": x, "qy": y2})

