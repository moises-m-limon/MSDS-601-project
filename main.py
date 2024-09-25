import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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

# Function to generate synthetic data
def generate_synthetic_data():
    n_samples = st.sidebar.slider("Number of Samples", 10, 100, 30)
    slope = st.sidebar.number_input("Slope", 1.0)
    intercept = st.sidebar.number_input("Intercept", 0.0)
    noise_level = st.sidebar.slider("Noise Level", 0.0, 10.0, 1.0)

    np.random.seed(42)
    x = np.random.rand(n_samples) * 10
    noise = np.random.normal(0, noise_level, n_samples)
    y = slope * x + intercept + noise

    return pd.DataFrame({'x': x, 'y': y})

# Function to calculate summary statistics
def calculate_summary_statistics(data, x_column, y_column):
    summary_stats = {
        "Mean": [data[x_column].mean(), data[y_column].mean()],
        "Median": [data[x_column].median(), data[y_column].median()],
        "Standard Deviation": [data[x_column].std(ddof=1), data[y_column].std(ddof=1)],
        "Min": [data[x_column].min(), data[y_column].min()],
        "Max": [data[x_column].max(), data[y_column].max()],
        "Count": [len(data[x_column]), len(data[y_column])],
        "Correlation": [data[x_column].corr(data[y_column])]
    }
    return pd.DataFrame(summary_stats, index=["X", "Y"]).T

# Main logic
data_option = st.sidebar.selectbox("Select Data Input Method", ["Upload CSV", "Manual Input"])
data = load_data(data_option)

if data is not None:
    st.write("### Data Preview:")
    st.dataframe(data)

    if data_option == "Upload CSV":
        x_column = st.sidebar.selectbox("Select X Column", data.columns)
        y_column = st.sidebar.selectbox("Select Y Column", data.columns)
    else:
        x_column, y_column = 'x', 'y'

    # Calculate and display summary statistics
    summary_df = calculate_summary_statistics(data, x_column, y_column)
    st.write("### Summary Statistics:")
    st.dataframe(summary_df)

    # Plotting with Plotly
    st.write("### Simple Linear Regression Plot:")

    # Create scatter plot
    fig = px.scatter(data, x=x_column, y=y_column, title='Simple Linear Regression', labels={x_column: x_column, y_column: y_column})

    # Fit linear regression model
    model = np.polyfit(data[x_column], data[y_column], 1)
    line = model[0] * data[x_column] + model[1]
    
    # Add line to plot
    fig.add_traces(go.Scatter(x=data[x_column], y=line, mode='lines', name='Regression Line', line=dict(color='red')))
    
    # Show the figure
    st.plotly_chart(fig)
