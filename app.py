import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

# Function to set the background image
def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded_string}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
                font-family: 'Arial', sans-serif;
            }}
            h1, h2 {{
                color: white;
                text-align: center;
                margin-bottom: 10px;
            }}
            .stButton>button {{
                background-color: black;
                color: white;
                font-size: 1.2rem;
                font-weight: bold;
                border-radius: 5px;
                padding: 10px 20px;
                cursor: pointer;
            }}
            .stButton>button:hover {{
                background-color: #555;
                color: #fff;
            }}
            .predicted-price-box {{
                background-color: rgba(0, 0, 0, 0.8);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                color: white;
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
            }}
            .predicted-price-box h2 {{
                color: #00ff00;
                font-size: 1.8rem;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Set background
set_background("laptop_image.jpg")

# Load the model
file1 = open('pipe.pkl', 'rb')
rf = pickle.load(file1)
file1.close()

# Load dataset
data = pd.read_csv("traineddata.csv")

# App title
st.markdown("<h1>Laptop Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Collapsible section: Laptop Specifications
with st.expander("💻 Laptop Specifications", expanded=True):
    st.markdown("<br>", unsafe_allow_html=True)
    # Row 1 for Laptop Specifications
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        company = st.selectbox('Select Brand', data['Company'].unique(), help="Choose the laptop's brand.")
        ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64], help="Select the RAM size.")
    with row1_col2:
        weight = st.number_input('Laptop Weight (in kg)', format="%.2f", help="Enter the weight of the laptop.")
        os = st.selectbox('Operating System', data['OpSys'].unique(), help="Select the operating system.")
    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2 for Laptop Specifications
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        type = st.selectbox('Laptop Type', data['TypeName'].unique(), help="Choose the type of laptop (e.g., Ultrabook, Gaming).")
    with row2_col2:
        laptop_usage = st.selectbox('Usage Type', ['Business', 'Gaming', 'Personal', 'Programming', 'Student'], help="Select the intended usage type.")
st.markdown("<br><br>", unsafe_allow_html=True)

# Collapsible section: Performance Specifications
with st.expander("⚙️ Performance Specifications", expanded=True):
    st.markdown("<br>", unsafe_allow_html=True)
    # Row 1 for Performance Specifications
    perf_row1_col1, perf_row1_col2 = st.columns(2)
    with perf_row1_col1:
        cpu = st.selectbox('Processor', data['CPU_name'].unique(), help="Select the CPU model.")
        touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'], help="Does the laptop have a touchscreen?")
    with perf_row1_col2:
        hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048], help="Select the HDD storage.")
        ips = st.selectbox('IPS Display', ['No', 'Yes'], help="Does the laptop have an IPS display?")
    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2 for Performance Specifications
    perf_row2_col1, perf_row2_col2 = st.columns(2)
    with perf_row2_col1:
        gpu = st.selectbox('GPU Brand', data['Gpu brand'].unique(), help="Select the GPU brand.")
    with perf_row2_col2:
        screen_size = st.number_input('Screen Size (in inches)', format="%.2f", key="performance_screen_size", help="Enter the screen size.")
st.markdown("<br><br>", unsafe_allow_html=True)

# Predict Button
if st.button('Predict Price', help="Click to predict the laptop price!"):
    # Prediction logic
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    resolution = '1920x1080'  # Example default resolution
    X_resolution = int(resolution.split('x')[0])
    Y_resolution = int(resolution.split('x')[1])
    ppi = ((X_resolution**2) + (Y_resolution**2))**0.5 / screen_size

    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, gpu, os])
    query = query.reshape(1, 12)

    prediction = int(np.exp(rf.predict(query)[0]))

    st.markdown(
        f"""
        <div class="predicted-price-box">
            <h2>Predicted Price</h2>
            <p>The estimated price for this laptop is between <strong>{prediction - 1000}₹</strong> and <strong>{prediction + 1000}₹</strong>.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
