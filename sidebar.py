import streamlit as st

def sidebar():
    n_samples = st.sidebar.slider("Number of Samples", 10, 100, 30)
    slope = st.sidebar.slider("Slope", 1, 100, 10 )
    col1, col2 = st.sidebar.columns(2)

    with col1:
        x_min = st.number_input("x_min", value=0)

    with col2:
        x_max = st.number_input("x_max", value=5)

    error_dist = st.sidebar.selectbox(
        "Select Distribution of Errors", ["Normal", "t", "Uniform", "log-Normal"]
    )
    noise_level = st.sidebar.slider("Noise Level", .1, 50.0, 15.)
    noise_type = st.sidebar.radio(
        "Select noise type:",
        ("Homoskedastic", "Heteroskedastic")
    )
    z_noise = st.sidebar.slider("z-axis noise Level", .1, 1000.0, 55.)

    return n_samples, slope, x_min, x_max, error_dist, noise_level, noise_type, z_noise
