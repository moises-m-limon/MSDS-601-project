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

# Title of the app
st.title("Simple Linear Regression Interactive App")

# TODO : FEATURE ERROR W/ DIFF DISTRIBUTIONS
# Function to generate synthetic data
def generate_synthetic_data():
    n_samples = st.sidebar.slider("Number of Samples", 10, 100, 30)
    slope = st.sidebar.number_input("Slope", 1.0)
    intercept = st.sidebar.number_input("Intercept", 0.0)

    ## generate new feature -- t, uniform

    drop_box = st.sidebar.selectbox(
        "Select Distribution", ["Normal", "t", "Uniform", "log-Normal"]
    )
    noise_level = st.sidebar.slider("Noise Level", 0.0, 10.0, 1.0)

    np.random.seed(42)  # setting randomized fn
    x = np.random.rand(n_samples) * 10

    # use distr from selectbox
    if drop_box == "Normal":
        noise = np.random.normal(0, noise_level, n_samples)
    elif drop_box == "Uniform":
        noise = np.random.uniform(-noise_level, noise_level, n_samples)
    elif drop_box == "t":
        noise = np.random.standard_t(df=10, size=n_samples) * noise_level
    elif drop_box == "log-Normal":
        noise = np.random.lognormal(x.mean(), x.std(), n_samples)

    y = slope * x + intercept + noise
    y2 = slope * (x**2) + intercept + noise
    return pd.DataFrame({"x": x, "y": y, "qx": x, "qy": y2})


# Function to calculate summary statistics
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
def generate_multicollinear_data(n=100):
    np.random.seed(42)
    x1 = np.random.uniform(1, 10, n)
    x2 = 2 * x1 + np.random.normal(0, 0.1, n)
    y = 3 * x1 + 4 * x2 + np.random.normal(0, 1, n)

    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


# Function to generate non-multicollinear data
def generate_non_multicollinear_data(n=100):
    np.random.seed(42)
    x1 = np.random.uniform(1, 10, n)
    x2 = np.random.uniform(1, 10, n)  # Independent of x1
    y = 3 * x1 + 4 * x2 + np.random.normal(0, 1, n)

    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def generate_data_v2(y_expression="x**2", n=100, xmin=-20, xmax=50, sigma=700):
    np.random.seed(42)
    x = np.linspace(xmin, xmax, n)

    # Evaluate y based on the expression
    y = eval(y_expression) + np.random.normal(0, sigma, n)

    return pd.DataFrame({"x": x, "y": y})


def display_data(df):
    st.write("### Generated Quadratic Data (y = x^2 + noise):")

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="x", y="y", ax=ax)
    ax.set_title("Scatter Plot of Quadratic Data")
    st.pyplot(fig)


# Main logic
data = generate_synthetic_data()


if data is not None:
    complete_index = data.notnull().all(axis=1)
    data = data[complete_index]
    x_column, y_column = "x", "y"

    st.write("### Simple Linear Regression Plot:")
    plot = (
        ggplot(data, aes(x=x_column, y=y_column))
        + geom_point(color="blue", size=2)
        + geom_smooth(method="lm", color="red", se=False)
        + labs(title="Simple Linear Regression", x=x_column, y=y_column)
    )

    plot_quad = (
        ggplot(data, aes(x="qx", y="qy"))
        + geom_point(color="blue", size=2)
        + geom_smooth(method="lm", color="red", se=False)
        # + geom_line(aes(y=data[x_column]**2), color="red", linetype='dashed')
        + labs(title="Quadratic Data (True vs Linear)", x="x", y="y")
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

    st.write("### Residual Plots:")

    col1, col2 = st.columns(2)
        # First column: Display the first plot
    with col1:
        gg1 = ss_decomp(data[x_column].values.reshape(-1, 1), data[y_column].values)
        st.pyplot(gg1.draw())

    # Second column: Display the quadratic plot
    with col2:
        gg2 = ss_decomp(
        data["qx"].values.reshape(-1, 1), data["qy"].values, include_quadratic=True)
        st.pyplot(gg2.draw())

    # Radio button for analysis type
    analysis_type = st.sidebar.radio(
        "Select Analysis Type", ["Summary Statistics", "ANOVA"]
    )

    # Streamlit App
    st.title("Multicollinear vs Non-Multicollinear Data Comparison")

    # Choose which dataset to display
    dataset_choice = st.sidebar.selectbox(
        "Select Dataset", ("Multicollinear Data", "Non-Multicollinear Data")
    )

    # Sidebar for customizing the data
    n = st.sidebar.slider(
        "Number of data points", min_value=50, max_value=500, value=100, step=10
    )

    # Generate the data based on user input
    # Generate both datasets based on user input
    df_multicollinear = generate_multicollinear_data(n)
    df_non_multicollinear = generate_non_multicollinear_data(n)

        # Display data
    st.write("Multicollinear Data vs. Non-Multicollinear Data")

    # Fit the MLR model and get the coefficients
    mlr_model = smf.ols("y ~ x1 + x2", data=df_multicollinear).fit()
    mlr_non_model = smf.ols("y ~ x1 + x2", data=df_non_multicollinear).fit()

    # Create 3D scatter plot for the selected dataset using Plotly
    fig1 = go.Figure()
    fig2 = go.Figure()

    # Scatter plot of the data points (Multicollinear Data)
    fig1.add_trace(
        go.Scatter3d(
            x=df_multicollinear["x1"],
            y=df_multicollinear["x2"],
            z=df_multicollinear["y"],
            mode="markers",
            marker=dict(size=5, opacity=0.8),
        )
    )

    # Scatter plot of the data points (Non-Multicollinear Data)
    fig2.add_trace(
        go.Scatter3d(
            x=df_non_multicollinear["x1"],
            y=df_non_multicollinear["x2"],
            z=df_non_multicollinear["y"],
            mode="markers",
            marker=dict(size=5, opacity=0.8),
        )
    )

    # Add a surface plot representing the regression plane for both datasets

    # Multicollinear Data
    x1_range_1 = np.linspace(df_multicollinear["x1"].min(), df_multicollinear["x1"].max(), 20)
    x2_range_1 = np.linspace(df_multicollinear["x2"].min(), df_multicollinear["x2"].max(), 20)
    x1_grid_1, x2_grid_1 = np.meshgrid(x1_range_1, x2_range_1)

    # Non-Multicollinear Data
    x1_range_2 = np.linspace(df_non_multicollinear["x1"].min(), df_non_multicollinear["x1"].max(), 20)
    x2_range_2 = np.linspace(df_non_multicollinear["x2"].min(), df_non_multicollinear["x2"].max(), 20)
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
        scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="y"),
        margin=dict(l=0, r=0, b=0, t=50),
    )

    fig2.update_layout(
        title="Non-Multicollinear Data",
        scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="y"),
        margin=dict(l=0, r=0, b=0, t=50),
    )

    # Display the plots in Streamlit
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1)
    with col2:
        st.plotly_chart(fig2)

    # Plotting for Linear Regression
    if analysis_type == "Summary Statistics":
        # Calculate and display summary statistics first
        summary_df = calculate_summary_statistics(data, x_column, y_column)
        st.write("### Summary Statistics:")
        st.dataframe(summary_df)

    # Display ANOVA table
    if analysis_type == "ANOVA":
        formula = f"{y_column} ~ {x_column}"
        model = ols(formula, data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        st.write("### ANOVA Table:")
        st.dataframe(anova_table)
