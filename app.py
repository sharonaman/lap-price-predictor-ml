import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

# Load the trained model
with open('pipe.pkl', 'rb') as file:
    rf = pickle.load(file)

# Load the dataset
data = pd.read_csv("traineddata.csv")

# Function to set background image and custom styling
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
            .stTitle {{
                color: white;
                text-align: center;
                font-size: 2.5rem;
                margin-bottom: 20px;
            }}
            .stButton>button {{
                background-color: #4CAF50;
                color: white;
                font-size: 1.2rem;
                font-weight: bold;
                border-radius: 10px;
                padding: 15px 40px;
                border: none;
                cursor: pointer;
            }}
            .stButton>button:hover {{
                background-color: #45a049;
            }}
            .predicted-price-box {{
                background-color: rgba(0, 0, 0, 0.7);
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                color: white;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
                margin-top: 20px;
            }}
            .predicted-price-box h2 {{
                font-size: 2rem;
                color: #4CAF50;
            }}
            </style>
            """, unsafe_allow_html=True
        )

# Set background image
set_background("laptop_image.jpg")

# Streamlit App Title
st.markdown("<h1 class='stTitle'>💻 Laptop Price Predictor</h1>", unsafe_allow_html=True)

# Single-column layout for inputs
st.subheader("Select Laptop Specifications")

# Collect user inputs with validation
company = st.selectbox('Select Laptop Brand', data['Company'].unique())
type = st.selectbox('Select Laptop Type', data['TypeName'].unique())
ram = st.selectbox('Select RAM Size (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
os = st.selectbox('Select Operating System', data['OpSys'].unique())
touchscreen = st.selectbox('Touchscreen Available?', ['No', 'Yes'])
cpu = st.selectbox('Select Processor', data['CPU_name'].unique())
weight = st.number_input('Enter Laptop Weight (in kg)', format="%.2f")
ips = st.selectbox('IPS Display?', ['No', 'Yes'])
screen_size = st.number_input('Enter Screen Size (in inches)', format="%.2f")
resolution = st.selectbox('Select Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
    '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
hdd = st.selectbox('Select HDD Size (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('Select SSD Size (in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('Select GPU Brand', data['Gpu brand'].unique())

# Predict Button with validation
if st.button('Predict Price', help="Click to predict the laptop price!"):
    error = False
    
    # Validation for weight and screen size
    if weight <= 0:
        st.error("Error: Weight should be greater than 0 kg.")
        error = True
        
    if screen_size <= 0:
        st.error("Error: Screen size should be greater than 0 inches.")
        error = True
    
    # Check for valid categorical selections
    if company not in data['Company'].unique():
        st.error("Error: Invalid laptop brand selected.")
        error = True
    
    if type not in data['TypeName'].unique():
        st.error("Error: Invalid laptop type selected.")
        error = True
    
    if cpu not in data['CPU_name'].unique():
        st.error("Error: Invalid processor selected.")
        error = True
    
    if gpu not in data['Gpu brand'].unique():
        st.error("Error: Invalid GPU brand selected.")
        error = True
    
    if os not in data['OpSys'].unique():
        st.error("Error: Invalid operating system selected.")
        error = True

    # If no errors, make prediction
    if not error:
        try:
            # Convert categorical inputs to numeric values
            touchscreen = 1 if touchscreen == 'Yes' else 0
            ips = 1 if ips == 'Yes' else 0

            # Extract resolution details for PPI calculation
            X_resolution = int(resolution.split('x')[0])
            Y_resolution = int(resolution.split('x')[1])
            ppi = ((X_resolution**2) + (Y_resolution**2))**0.5 / screen_size

            # Create query array
            query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
            query = query.reshape(1, -1)  # Reshape dynamically to avoid size mismatch

            # Make prediction
            prediction = int(np.exp(rf.predict(query)[0]))
            st.markdown(
                f"""
                <div class="predicted-price-box">
                    <h2>Predicted Price</h2>
                    <p>The estimated price for this laptop is between <strong>₹{prediction - 1000}</strong> and <strong>₹{prediction + 1000}</strong>.</p>
                </div>
                """, unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error occurred during prediction: {e}")
