import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Customer Churn Prediction", page_icon="🔮", layout="centered")

st.title("🔮 Customer Churn Prediction App")

# 1. LOAD ASSETS (WITHOUT CACHE TO PREVENT OLD FILE ISSUES)
try:
    with open('preprocessing_assets.pkl', 'rb') as f:
        assets = pickle.load(f)
    with open('mlp_churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    label_encoders = assets['label_encoders']
    scaler = assets['scaler']
    numerical_cols = assets['numerical_columns']
    categorical_cols = assets['categorical_columns']
    feature_names = assets['feature_names']
except Exception as e:
    st.error(f"Error loading pickle files: {e}")
    st.stop()

# 2. QUICK TEST BUTTONS TO VERIFY DYNAMIC PROBABILITIES
st.subheader("⚡ Quick Test Presets")
preset_col1, preset_col2 = st.columns(2)

if 'age_val' not in st.session_state:
    st.session_state.age_val = 35
    st.session_state.gender_val = 'Female'
    st.session_state.tenure_val = 12
    st.session_state.usage_val = 15
    st.session_state.calls_val = 2
    st.session_state.delay_val = 5
    st.session_state.sub_val = 'Basic'
    st.session_state.contract_val = 'Monthly'
    st.session_state.spend_val = 500.0
    st.session_state.last_val = 10

if preset_col1.button("🔴 Load High Risk Customer Sample"):
    st.session_state.age_val = 51
    st.session_state.gender_val = 'Female'
    st.session_state.tenure_val = 7
    st.session_state.usage_val = 4
    st.session_state.calls_val = 7
    st.session_state.delay_val = 25
    st.session_state.sub_val = 'Basic'
    st.session_state.contract_val = 'Monthly'
    st.session_state.spend_val = 150.0
    st.session_state.last_val = 28
    st.rerun()

if preset_col2.button("🟢 Load Low Risk Customer Sample"):
    st.session_state.age_val = 26
    st.session_state.gender_val = 'Male'
    st.session_state.tenure_val = 51
    st.session_state.usage_val = 19
    st.session_state.calls_val = 1
    st.session_state.delay_val = 0
    st.session_state.sub_val = 'Premium'
    st.session_state.contract_val = 'Annual'
    st.session_state.spend_val = 800.0
    st.session_state.last_val = 5
    st.rerun()

# 3. FORM INPUTS
st.subheader("📋 Customer Details Input")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=st.session_state.age_val)
    gender = st.selectbox("Gender", options=label_encoders['Gender'].classes_, index=list(label_encoders['Gender'].classes_).index(st.session_state.gender_val))
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=st.session_state.tenure_val)
    usage_frequency = st.number_input("Usage Frequency", min_value=0, max_value=100, value=st.session_state.usage_val)
    support_calls = st.number_input("Support Calls", min_value=0, max_value=50, value=st.session_state.calls_val)

with col2:
    payment_delay = st.number_input("Payment Delay (Days)", min_value=0, max_value=60, value=st.session_state.delay_val)
    subscription_type = st.selectbox("Subscription Type", options=label_encoders['Subscription Type'].classes_, index=list(label_encoders['Subscription Type'].classes_).index(st.session_state.sub_val))
    contract_length = st.selectbox("Contract Length", options=label_encoders['Contract Length'].classes_, index=list(label_encoders['Contract Length'].classes_).index(st.session_state.contract_val))
    total_spend = st.number_input("Total Spend ($)", min_value=0.0, max_value=10000.0, value=float(st.session_state.spend_val))
    last_interaction = st.number_input("Last Interaction (Days Ago)", min_value=0, max_value=30, value=st.session_state.last_val)

# 4. PREDICTION LOGIC
if st.button("🚀 Predict Churn Risk", use_container_width=True):
    input_dict = {
        'Age': age, 'Gender': gender, 'Tenure': tenure,
        'Usage Frequency': usage_frequency, 'Support Calls': support_calls,
        'Payment Delay': payment_delay, 'Subscription Type': subscription_type,
        'Contract Length': contract_length, 'Total Spend': total_spend,
        'Last Interaction': last_interaction
    }
    
    input_df = pd.DataFrame([input_dict])

    # Transform categoricals
    for col in categorical_cols:
        input_df[col] = label_encoders[col].transform(input_df[col])

    # Ensure column order matches training set
    input_df = input_df[feature_names]

    # Transform numericals
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Predict
    probs = model.predict_proba(input_df)[0]
    stay_prob = probs[0] * 100
    churn_prob = probs[1] * 100
    pred_class = model.predict(input_df)[0]

    st.markdown("---")
    st.subheader("📊 Prediction Results")

    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Retention Rate (Stay)", f"{stay_prob:.2f}%")
    res_col2.metric("Churn Risk Probability", f"{churn_prob:.2f}%")

    if pred_class == 1:
        st.error(f"⚠️ High Risk: Customer likely to Churn! (Risk: {churn_prob:.2f}%)")
    else:
        st.success(f"✅ Low Risk: Customer likely to Stay. (Risk: {churn_prob:.2f}%)")