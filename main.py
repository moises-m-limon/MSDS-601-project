import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_point, geom_smooth, labs, geom_line
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from ss_decomp import ss_decomp
import plotly.graph_objs as go
from resid_plot import display_resid
from resid_hist import display_resid_histogram
from generate_data_functions import * 

# Title of the app
st.header("Data Doctor: Diagnosing regression problems")

# sidebar
n_samples = st.sidebar.slider("Number of Samples", 10, 100, 30)
slope = st.sidebar.slider("Slope", 1, 100, 10)
intercept = st.sidebar.slider("Intercept", -100, 100, 10)
col1, col2 = st.sidebar.columns(2)
with col1:
    x_min = st.number_input("x_min", value=0)

with col2:
    x_max = st.number_input("x_max", value=5)

error_dist = st.sidebar.selectbox(
    "Select Distribution of Errors", ["Normal", "t", "Uniform"]
)
noise_level = st.sidebar.slider("Noise Level", .1, 50.0, 15.)
noise_type = st.sidebar.radio(
    "Select noise type:",
    ("Homoskedastic", "Heteroskedastic")
)



# Main logic
data = generate_synthetic_data(n_samples, intercept, slope, x_min, x_max, error_dist, noise_level, noise_type)

complete_index = data.notnull().all(axis=1)
data = data[complete_index]
x_column, y_column = "x", "y"

st.write('What if we try to fit a linear model to non-linear data?')
plot = (
    ggplot(data, aes(x=x_column, y=y_column))
    + geom_point(color="blue", size=2)
    + geom_smooth(method="lm", color="red", se=False)
    + labs(title="X & Y have a linear relationship", x=x_column, y=y_column)
)

plot_quad = (
    ggplot(data, aes(x="qx", y="qy"))
    + geom_point(color="blue", size=2)
    + geom_smooth(method="lm", color="red", se=False)
    # + geom_line(aes(y=data[x_column]**2), color="red", linetype='dashed')
    + labs(title="X & Y have a non-linear relationship", x="x", y="y")
)

col1, col2 = st.columns(2)

# First column: Display the first plot
with col1:
    fig = plot.draw()
    plt.close(fig)
    st.pyplot(fig)

# Second column: Display the quadratic plot
with col2:
    fig_quad = plot_quad.draw()
    plt.close(fig_quad)
    st.pyplot(fig_quad)

st.write("Solution: Transformation of one of our variables")

transform = st.selectbox(
    "Select a transformation for x:",
    ["y = log(x)", "y = e^x", "y = x^2", "y = x^(1/2)"]
)

# Apply the transformation based on the selection
transformed_data = data.copy()
if transform == "y = x^2":
    transformed_data['qy'] = np.sqrt(transformed_data['qy'])
elif transform == "y = log(x)":
    # Ensure all values are positive before applying log
    transformed_data = transformed_data[transformed_data['qy'] > 0]
    transformed_data['qy'] = np.log(transformed_data['qy'])
elif transform == "y = e^x":
    transformed_data['qy'] = np.log(transformed_data['qy'])
elif transform == "y = x^(1/2)":
    transformed_data['qy'] = transformed_data['qy'] ** 2

# Now create the ggplot with the transformed data
plot_transform = (
    ggplot(transformed_data, aes(x="qx", y="qy"))
    + geom_point(color="blue", size=2)
    + geom_smooth(method="lm", color="red", se=False)
    + labs(title=f"Transformation: {transform}", x="x", y="y")
)

col1, col2 = st.columns(2)
    # First column: Display the original plot
with col1:
    # gg1 = ss_decomp(data[x_column].values.reshape(-1, 1), data[y_column].values)
    st.pyplot(plot_quad.draw())

# Second column: Display the quadratic res plot
with col2:
    st.pyplot(plot_transform.draw())

st.write("### Residual Plots:")
st.write('These plots are a great way to see check for heteroskedasticity.')

col1, col2 = st.columns(2)
    # First column: Display the res plot
with col1:
    resid_plot =  display_resid(data['x'], data['y'],"X & Y have a linear relationship")
    st.pyplot(resid_plot.draw())

# Second column: Display the quadratic res plot
with col2:
    q_resid_plot =  display_resid(data['x'], data['qy'], "X & Y have a non-linear relationship")
    st.pyplot(q_resid_plot.draw())

st.write("### Residual Histogram:")



resid_hist =  display_resid_histogram(data['x'], data['y'],"X & Y have a linear relationship")
st.pyplot(resid_hist.draw())

# Radio button for analysis type
# analysis_type = st.sidebar.radio(
#     "Select Analysis Type", ["Summary Statistics", "ANOVA"]
# )
# # Plotting for Linear Regression
# if analysis_type == "Summary Statistics":
#     # Calculate and display summary statistics first
#     summary_df = calculate_summary_statistics(data, x_column, y_column)
#     st.write("### Summary Statistics:")
#     st.dataframe(summary_df)

# # Display ANOVA table
# if analysis_type == "ANOVA":
#     formula = f"{y_column} ~ {x_column}"
#     model = ols(formula, data=data).fit()
#     anova_table = sm.stats.anova_lm(model, typ=2)
#     st.write("### ANOVA Table:")
#     st.dataframe(anova_table)
