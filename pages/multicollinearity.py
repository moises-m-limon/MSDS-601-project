import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_point, geom_smooth, labs, geom_line
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
import plotly.graph_objs as go
from generate_data_functions import * 
from sidebar import sidebar

# Streamlit App 
st.title("Multicollinearity")

# Generate the data based on user input

delta_z = st.sidebar.slider("change height", min_value=-10., max_value=10., value=0., step=.1)
# along_line = st.sidebar.slider("move along line", min_value=, max_value=10., value=0., step=.1)

delta_inf = st.sidebar.slider("shift influence", min_value=-3., max_value=3., value=0.0, step =.1)
n_samples = st.sidebar.slider("n", 10, 100, 30)
# slope = st.sidebar.slider("Slope", 0, 100, 1)
# intercept = st.sidebar.slider("Intercept", -100, 100, 10)

# error_dist = st.sidebar.selectbox(
#     "Select Distribution of Errors", ["Normal", "t", "Uniform"]
# )
noise_level = st.sidebar.slider("X,Y noise level", 0.0, 2.0, 0.1)
z_noise = st.sidebar.slider("Z noise level", 0., 10., .1)
noise_type = st.sidebar.radio(
    "Select noise type:",
    ("Homoskedastic", "Heteroskedastic")
)

df_multicollinear = generate_multicollinear_data(n_samples, noise_level, z_noise, noise_type)
df_non_multicollinear = generate_non_multicollinear_data(z_noise, n = n_samples)


def insert_non_multicolinnear_point(df):
    new_row = {"x": np.mean(df['x']) -delta_inf, "y": np.mean(df['y']) +delta_inf, "z": np.mean(df['z']) + delta_z}
    df.loc[len(df)] = new_row
    return df

df_multicollinear = insert_non_multicolinnear_point(df_multicollinear)
df_non_multicollinear = insert_non_multicolinnear_point(df_non_multicollinear)


# Fit the MLR model and get the coefficients
mlr_model = smf.ols("z ~ x + y", data=df_multicollinear).fit()
mlr_non_model = smf.ols("z ~ x + y", data=df_non_multicollinear).fit()

# Create 3D scatter plot for the selected dataset using Plotly
fig1 = go.Figure()
fig2 = go.Figure()

# Scatter plot of the data points (Multicollinear Data)
fig1.add_trace(
    go.Scatter3d(
        x=df_multicollinear["x"],
        y=df_multicollinear["y"],
        z=df_multicollinear["z"],
        mode="markers",
        marker=dict(size=5, opacity=0.8),
    )
)

# Scatter plot of the data points (Non-Multicollinear Data)
fig2.add_trace(
    go.Scatter3d(
        x=df_non_multicollinear["x"],
        y=df_non_multicollinear["y"],
        z=df_non_multicollinear["z"],
        mode="markers",
        marker=dict(size=5, opacity=0.8),
    )
)

# Add a surface plot representing the regression plane for both datasets

# Multicollinear Data
x1_range_1 = np.linspace(df_multicollinear["x"].min(), df_multicollinear["x"].max(), 20)
x2_range_1 = np.linspace(df_multicollinear["y"].min(), df_multicollinear["y"].max(), 20)
x1_grid_1, x2_grid_1 = np.meshgrid(x1_range_1, x2_range_1)

# Non-Multicollinear Data
x1_range_2 = np.linspace(df_non_multicollinear["x"].min(), df_non_multicollinear["x"].max(), 20)
x2_range_2 = np.linspace(df_non_multicollinear["y"].min(), df_non_multicollinear["y"].max(), 20)
x1_grid_2, x2_grid_2 = np.meshgrid(x1_range_2, x2_range_2)

# Get regression plane parameters
b0_1, b1_1, b2_1 = mlr_model.params
b0_2, b1_2, b2_2 = mlr_non_model.params
y_grid_1 = b0_1 + b1_1 * x1_grid_1 + b2_1 * x2_grid_1
y_grid_2 = b0_2 + b1_2 * x1_grid_2 + b2_2 * x2_grid_2

# Surface plot of the fitted plane
fig1.add_trace(go.Surface(x=x1_grid_1, y=x2_grid_1, z=y_grid_1, opacity=0.5))
fig2.add_trace(go.Surface(x=x1_grid_2, y=x2_grid_2, z=y_grid_2, opacity=0.5))

# Layout settings
fig1.update_layout(
    title="Multicollinear Data",
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
    margin=dict(l=0, r=0, b=0, t=50),
)

fig2.update_layout(
    title="Non-Multicollinear Data",
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
    margin=dict(l=0, r=0, b=0, t=50),
)
st.plotly_chart(fig1)

st.plotly_chart(fig2)
