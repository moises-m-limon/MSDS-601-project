import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_point, geom_smooth, labs, geom_histogram
import statsmodels.api as sm
from statsmodels.formula.api import ols
from ss_decomp import ss_decomp

# Title of the app
st.title("Simple Linear Regression Interactive App")


# Function to load and clean data
def load_data(data_option):
    if data_option == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            if data.empty:
                st.error("The uploaded CSV file is empty.")
                return None
            st.success("Data loaded successfully.")
            return data
    else:
        return generate_synthetic_data()


# TODO : FEATURE ERROR W/ DIFF DISTRIBUTIONS
# Function to generate synthetic data
def generate_synthetic_data():
    n_samples = st.sidebar.slider("Number of Samples", 10, 100, 30)
    slope = st.sidebar.number_input("Slope", 1.0)
    intercept = st.sidebar.number_input("Intercept", 0.0)

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
    return pd.DataFrame({"x": x, "y": y}), noise

#fn to plot histogram of errors 
def plot_error_histogram(df, noise):
    print(noise)
    plot = (ggplot(df, aes(x='noise')) +
            geom_histogram(binwidth=0.5, fill='skyblue', color='black') +
            labs(title='Error Histogram', x='Error (Noise)', y='Frequency'))
    return plot

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


# Main logic
data_option = st.sidebar.radio(
    "Select Data Input Method", ["Generate Data", "Upload CSV"]
)
data, noise = load_data(data_option)


if data is not None:
    complete_index = data.notnull().all(axis=1)
    data = data[complete_index]
    if data_option == "Upload CSV":
        x_column = st.sidebar.selectbox("Select X Column", data.columns)
        y_column = st.sidebar.selectbox("Select Y Column", data.columns)
    else:
        x_column, y_column = "x", "y"

    st.write("### Simple Linear Regression Plot:")
    plot = (
        ggplot(data, aes(x=x_column, y=y_column))
        + geom_point(color="blue", size=2)
        + geom_smooth(method="lm", color="red", se=False)
        + labs(title="Simple Linear Regression", x=x_column, y=y_column)
    )

    fig = plot.draw()
    plt.close(fig)
    st.pyplot(fig)

    st.write("### Residual Plots:")
    gg1 = ss_decomp(data[x_column].values.reshape(-1, 1), data[y_column].values)
    st.pyplot(gg1.draw())

    st.write("### Error Histogram:")
    st.pyplot(plot_error_histogram(data, noise).draw())

    st.write("### Data Preview:")
    st.dataframe(data)
    # Radio button for analysis type
    analysis_type = st.sidebar.radio(
        "Select Analysis Type", ["Summary Statistics", "ANOVA"]
    )

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
