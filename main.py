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

# Title of the app
st.header("Data Doctor: Diagnosing regression problems")

# Function to generate synthetic data
def generate_synthetic_data():
    n_samples = st.sidebar.slider("Number of Samples", 10, 100, 30)
    slope = st.sidebar.slider("Slope", 1, 100, 10 )
    col1, col2 = st.sidebar.columns(2)

    # Add inputs to the sidebar columns
    with col1:
        x_min = st.number_input("x_min", value=0)

    with col2:
        x_max = st.number_input("x_max", value=5)

    # intercept = st.sidebar.number_input("Intercept", 0.0)

    ## generate new feature -- t, uniform

    drop_box = st.sidebar.selectbox(
        "Select Distribution of Errors", ["Normal", "t", "Uniform", "log-Normal"]
    )
    noise_level = st.sidebar.slider("Noise Level", 5.0, 50.0, 15.)
    noise_type = st.sidebar.radio(
        "Select noise type:",
        ("Homoskedastic", "Heteroskedastic")
    )

    np.random.seed(42)  # setting randomized fn
    x = np.random.uniform(low=x_min, high=x_max, size=n_samples)
    # use distr from selectbox
    if drop_box == "Normal":
        noise = np.random.normal(0, noise_level, n_samples)
    elif drop_box == "Uniform":
        noise = np.random.uniform(-noise_level, noise_level, n_samples)
    elif drop_box == "t":
        noise = np.random.standard_t(df=10, size=n_samples) * noise_level
    elif drop_box == "log-Normal":
        noise = np.random.lognormal(x.mean(), x.std(), n_samples)

    if noise_type == "Heteroskedastic":
        noise = noise * x/2

    y = slope * x + noise
    y2 = slope * (x**2) + noise

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

# Main logic
data = generate_synthetic_data()


if data is not None:
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
    analysis_type = st.sidebar.radio(
        "Select Analysis Type", ["Summary Statistics", "ANOVA"]
    )

    # Streamlit App
    st.title("Multicollinearity")

    # # Choose which dataset to display
    # dataset_choice = st.sidebar.selectbox(
    #     "Select Dataset", ("Multicollinear Data", "Non-Multicollinear Data")
    # )

    # Sidebar for customizing the data
    # n = st.sidebar.slider(
    #     "Number of data points", min_value=50, max_value=500, value=100, step=10
    # )

    # Generate the data based on user input
    # Generate both datasets based on user input
    df_multicollinear = generate_multicollinear_data(100)
    df_non_multicollinear = generate_non_multicollinear_data(100)

        # Display data


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
