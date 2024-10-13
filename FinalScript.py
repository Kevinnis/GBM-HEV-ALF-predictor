import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sksurv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.ensemble.survival_loss import DummySurvivalEstimator
import shap
import pickle

# Load the model
model = joblib.load("gbm_mod1.pkl")

# Define feature names
feature_names = ["NEU","MONO","INR","AST","ALB","TBIL","UREA"]

# Streamlit user interface
st.set_page_config(
    page_title="HEV-ALF Risk Predictor",
    page_icon=":microbe:",  # Adding a custom icon (microbe emoji)
    layout="centered",  # Center the content
)

st.title("HEV-ALF Risk Predictor")
st.caption('This online machine-learning prediction tool was developed to predict the risk of hepatitis E virus-related acute liver failure among hospitalized patients with acute hepatitis E')

# Customizing the appearance of the input form using streamlit's markdown styling
st.markdown("""
    <style>
        .stNumberInput {
            width: 100%;
            padding: 10px;
            font-size: 16px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
    </style>
""", unsafe_allow_html=True)


# Numerical input
NEU = st.number_input("Neutrophil count (10^9/L)", min_value=0.0, max_value=10000.0, format="%.2f", key="NEU")
MONO = st.number_input("Monocyte count (10^9/L)", min_value=0.0, max_value=10000.0, format="%.2f", key="MONO")
INR = st.number_input("International normalized ratio", min_value=0.0, max_value=100.0, format="%.2f", key="INR")
AST = st.number_input("Aspartate aminotransferase (U/L)", min_value=0.0, max_value=10000.0, format="%.2f", key="AST")
ALB = st.number_input("Albumin (g/L)", min_value=0.0, max_value=10000.0, format="%.2f", key="ALB")
TBIL = st.number_input("Total bilirubin (μmol/L)", min_value=0.0, max_value=10000.0, format="%.2f", key="TBIL")
UREA = st.number_input("Urea (mmol/L)", min_value=0.0, max_value=10000.0, format="%.2f", key="UREA")

feature_values = [NEU, MONO, INR, AST, ALB, TBIL, UREA]
features = np.array([feature_values])

# Center the predict button
st.markdown("""
    <style>
    div.stButton > button:first-child {
        display: block;
        margin: 0 auto;
    }
    </style>""", unsafe_allow_html=True)

# Predict button
if st.button("Predict"):    
    # Predict risk score
    risk_score = model.predict(features)[0]
    
    st.markdown("<h3 style='color: black; font-weight: bold;'>Prediction Results</h3>", unsafe_allow_html=True)

    # Display Risk Score
    st.markdown(f"<h3 style='text-align: center;'>Risk Score: {risk_score:.4f}</h3>", unsafe_allow_html=True)

    # Display HEV-ALF onset risk based on threshold
    if risk_score >= 0.2163745:
        st.markdown("<h3 style='text-align: center; color: red;'>7-day HEV-ALF onset risk: High-risk</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='text-align: center; color: green;'>7-day HEV-ALF onset risk: Low-risk</h3>", unsafe_allow_html=True)

    if risk_score >= 0.2163745:
        st.markdown("<h3 style='text-align: center; color: red;'>14-day HEV-ALF onset risk: High-risk</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='text-align: center; color: green;'>14-day HEV-ALF onset risk: Low-risk</h3>", unsafe_allow_html=True)

    if risk_score >= 0.2163745:
        st.markdown("<h3 style='text-align: center; color: red;'>28-day HEV-ALF onset risk: High-risk</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='text-align: center; color: green;'>28-day HEV-ALF onset risk: Low-risk</h3>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='color: black; font-weight: bold;'>Prediction Interpretation</h3>", unsafe_allow_html=True)
    st.caption('The explanation for this prediction is shown below. Please note the prediction results should be interpreted by medical professionals.')
    
    # Compute SHAP values
    explainer = joblib.load('shap_explainer.pkl')
    shap_values = explainer(features)
    
    # Create a figure for the SHAP force plot
    features_df = pd.DataFrame([feature_values], columns=feature_names)
    shap.plots.force(shap_values.base_values,
                 shap_values.values[0],
                 pd.DataFrame([features_df.iloc[0].values], columns=features_df.columns),matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")

st.caption('Version: 20241014[This is currently a demo version for review]')
st.caption('Contact: wangjienjmu@126.com')
