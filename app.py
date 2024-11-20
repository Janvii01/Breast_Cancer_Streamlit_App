import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the scaler and models
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

models = {
    'Logistic Regression': 'best_lr_model.pkl',
    'Decision Tree': 'best_dt_model.pkl',
    'Random Forest': 'best_rf_model.pkl',
    'ANN': 'best_ann_model.pkl'
}

# Function to load the model
def load_model(model_name):
    with open(models[model_name], 'rb') as f:
        model = pickle.load(f)
    return model

# Streamlit UI
st.title("Breast Cancer Prediction App")

# Sidebar for user input
st.sidebar.header('User Input Features')

def user_input_features():
    """
    Function to get user input from the sidebar.
    Returns a DataFrame of user input features.
    """
    radius_mean = st.sidebar.slider('Mean Radius', 6.0, 30.0, 14.0)
    texture_mean = st.sidebar.slider('Mean Texture', 9.0, 40.0, 19.0)
    perimeter_mean = st.sidebar.slider('Mean Perimeter', 50.0, 200.0, 100.0)
    area_mean = st.sidebar.slider('Mean Area', 100.0, 3000.0, 500.0)
    smoothness_mean = st.sidebar.slider('Mean Smoothness', 0.05, 0.15, 0.1)
    
    # Create a dictionary with the input values
    user_data = {
        'radius_mean': radius_mean,
        'texture_mean': texture_mean,
        'perimeter_mean': perimeter_mean,
        'area_mean': area_mean,
        'smoothness_mean': smoothness_mean
    }
    
    # Convert the dictionary to a pandas DataFrame
    features = pd.DataFrame(user_data, index=[0])
    
    return features

# Get user input data
user_input = user_input_features()

# Display the user input data
st.subheader('User Input Data')
st.write(user_input)

# Model selection
model_name = st.sidebar.selectbox("Select a Model", options=list(models.keys()))

# Load the selected model
model = load_model(model_name)

# Scale the user input data using the loaded scaler
user_input_scaled = scaler.transform(user_input)

# Make prediction using the selected model
prediction = model.predict(user_input_scaled)
prediction_proba = model.predict_proba(user_input_scaled)

# Display the prediction result
if prediction == 0:
    st.write(f"The model predicts that the tumor is **benign** (0) using {model_name}.")
else:
    st.write(f"The model predicts that the tumor is **malignant** (1) using {model_name}.")

# Display the prediction probabilities
st.subheader('Prediction Probability')
st.write(f"Probability of benign: {prediction_proba[0][0]:.2f}")
st.write(f"Probability of malignant: {prediction_proba[0][1]:.2f}")
