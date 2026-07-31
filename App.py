import streamlit as st
import joblib

model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Credit Risk Prediction")

st.success("Model Loaded Successfully ✅")


import numpy as np

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(
    page_title="Creditworthiness Prediction",
    page_icon="💳",
    layout="wide"
)

# --------------------------
# Custom CSS
# --------------------------
st.markdown("""
<style>

.main{
    background-color:#F5F7FA;
}

.title{
    text-align:center;
    color:#003366;
    font-size:42px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}

div[data-testid="stButton"] button{
    background:#0066CC;
    color:white;
    border-radius:10px;
    height:55px;
    width:100%;
    font-size:20px;
    font-weight:bold;
}

div[data-testid="stButton"] button:hover{
    background:#004999;
}

.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 0px 15px rgba(0,0,0,0.15);
}

</style>
""", unsafe_allow_html=True)

# --------------------------
# Load Model
# --------------------------

model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")

# --------------------------
# Title
# --------------------------

st.markdown('<p class="title">💳 Creditworthiness Prediction System</p>', unsafe_allow_html=True)

st.markdown('<p class="subtitle">Machine Learning Based Loan Risk Assessment</p>', unsafe_allow_html=True)

st.write("")

# --------------------------
# Sidebar
# --------------------------

st.sidebar.title("Navigation")

st.sidebar.info(
"""
This application predicts whether a customer
is creditworthy using a trained Machine Learning model.
"""
)

st.sidebar.success("Decision Tree Classifier")

# --------------------------
# Input Section
# --------------------------

col1,col2=st.columns(2)

with col1:

    st.markdown("### 👤 Customer Information")

    clientid=st.number_input(
        "Client ID",
        min_value=1,
        value=1
    )

    age=st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

with col2:

    st.markdown("### 💰 Financial Information")

    income=st.number_input(
        "Annual Income",
        min_value=0.0,
        value=50000.0
    )

    loan=st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=10000.0
    )

st.write("")

# --------------------------
# Prediction Button
# --------------------------

if st.button("🔍 Predict Creditworthiness"):

    input_data=np.array([[clientid,
                          income,
                          age,
                          loan]])

    input_scaled=scaler.transform(input_data)

    prediction=model.predict(input_scaled)

    probability=model.predict_proba(input_scaled)

    risk=probability[0][1]
    approval=probability[0][0]

    st.write("---")

    st.header("Prediction Result")

    if prediction[0]==0:

        st.success("✅ Customer is Creditworthy")

    else:

        st.error("❌ Customer is High Credit Risk")

    st.write("")

    st.subheader("Approval Probability")

    st.progress(float(approval))

    st.write(f"**{approval*100:.2f}%**")

    st.subheader("Risk Probability")

    st.progress(float(risk))

    st.write(f"**{risk*100:.2f}%**")

    st.write("")

    st.metric(
        label="Predicted Class",
        value=int(prediction[0])
    )

# --------------------------
# Footer
# --------------------------

st.write("---")

st.caption(
"""
Developed by Muhammad Tayyab

Machine Learning Internship Project
"""
)