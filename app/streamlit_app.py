import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os

# Page config
st.set_page_config(
    page_title="ChurnGuard ML",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS for consistent typography
st.markdown("""
<style>
            
    /* Remove top padding */
    .block-container {
        padding-top: 2rem !important;
    }
            
    /* Main title */
    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Subtitle below title */
    .subtitle {
        font-size: 1.1rem !important;
        color: #666 !important;
        margin-top: -10px !important;
    }
    
    /* Section headers (st.subheader) */
    h2 {
        font-size: 1.6rem !important;
        margin-top: 1rem !important;
    }
    
    /* Subsection headers (###) */
    h3 {
        font-size: 1.3rem !important;
    }
    
    /* Small headers (#####) */
    h4, h5 {
        font-size: 1.1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Body text and lists */
    p, li {
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }
    
    /* Make bold text slightly larger */
    strong {
        font-size: 1.05rem !important;
    }
    
    /* Tab text */
    button[data-baseweb="tab"] {
        font-size: 1rem !important;
    }
    
    /* Expander headers */
    [data-testid="stExpander"] summary {
        font-size: 1rem !important;
    }
    
    /* Input labels */
    label {
        font-size: 0.95rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('../models/churnguard_model.joblib')
    label_encoders = joblib.load('../models/label_encoders.joblib')
    feature_names = joblib.load('../models/feature_names.joblib')
    return model, label_encoders, feature_names

try:
    model, label_encoders, feature_names = load_model()
except:
    st.error("Could not load model files. Please run the notebook first!")
    st.stop()

# Title
st.title("ChurnGuard ML")
st.markdown("<p class='subtitle'>Customer Churn Prediction System</p>", unsafe_allow_html=True)
st.markdown("---")

# TOP SECTION - Key Inputs + Quick Predict
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 0.7])

with col1:
    tenure = st.number_input("Tenure (months)", 0, 72, 18, key="tenure_top")
    
with col2:
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 85.0, step=5.0, key="charges_top")
    
with col3:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="contract_top")
    
with col4:
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], key="internet_top")

with col5:
    st.markdown("<br>", unsafe_allow_html=True)
    quick_predict = st.button("Predict Churn Risk", type="primary", use_container_width=True, key="quick_predict_btn")

st.markdown("---")

# TWO COLUMN LAYOUT
left_col, right_col = st.columns([0.7, 2.1])

with left_col:
    st.subheader("Customer Details")
    
    # Demographics (compact)
    with st.expander("Demographics", expanded=True):
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
    
    # Payment Info (compact)
    with st.expander("Payment & Billing", expanded=True):
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", 
                                      ["Electronic check", "Mailed check", 
                                       "Bank transfer (automatic)", "Credit card (automatic)"])
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 730.0, step=100.0)
    
    # Services (compact)
    with st.expander("Services", expanded=False):
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    # Predict button (prominent)
    st.markdown("---")
    detailed_predict = st.button("Predict Churn Risk", type="primary", use_container_width=True, key="detailed_predict_btn")

# Check if either predict button was clicked
predict_btn = quick_predict or detailed_predict

with right_col:
    if predict_btn:
        # Create dataframe
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [senior_citizen],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })
        
        # Add engineered feature
        input_data['customer_value'] = input_data['MonthlyCharges'] * input_data['tenure']
        
        # Encode
        for col in label_encoders.keys():
            if col in input_data.columns:
                try:
                    input_data[col] = label_encoders[col].transform(input_data[col])
                except:
                    input_data[col] = 0
        
        # Ensure correct order
        input_data = input_data[feature_names]
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        churn_prob = probability[1]
        
        # Results Section
        st.subheader("Prediction Results")
        
        # Metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.markdown("**Churn Probability**")
            st.markdown(f"<h3 style='margin-top: -15px; font-size: 1.5rem;'><b>{churn_prob:.1%}</b></h3>", 
                unsafe_allow_html=True)

        with metric_col2:
            if churn_prob > 0.7:
                risk_level, risk_color = "High", "red"
            elif churn_prob > 0.4:
                risk_level, risk_color = "Medium", "orange"
            else:
                risk_level, risk_color = "Low", "green"
            
            st.markdown("**Risk Level**")
            st.markdown(f"<h3 style='color: {risk_color}; margin-top: -15px; font-size: 1.5rem;'><b>{risk_level.upper()}</b></h3>", 
                        unsafe_allow_html=True)

        with metric_col3:
            prediction_text = "Will Churn" if prediction == 1 else "Will Stay"
            prediction_color = "red" if prediction == 1 else "green"
            
            st.markdown("**Prediction**")
            st.markdown(f"<h3 style='color: {prediction_color}; margin-top: -15px; font-size: 1.5rem;'><b>{prediction_text.upper()}</b></h3>", 
                        unsafe_allow_html=True)
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = churn_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Risk Score", 'font': {'size': 20}},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100], 'tickfont': {'size': 14}},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            },
            number = {'font': {'size': 60}}
        ))
        
        fig.update_layout(height=450, font={'size': 20}, margin=dict(t=60, b=50, l=50, r=50))
        st.plotly_chart(fig, use_container_width=True)
        # Add spacing before Retention Strategy
        st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)  # ← NEW LINE
                
        # Recommendations
        st.subheader("Retention Strategy")

        if churn_prob > 0.7:
            st.markdown("""
            <div style='background-color: #ffebee; padding: 10px; border-left: 5px solid #d32f2f; border-radius: 5px;'>
                <h4 style='color: #d32f2f; margin-top: 0; font-size: 1.3rem;'>High Risk - Immediate Action Required</h4>
            </div>
            """, unsafe_allow_html=True)
            
            rec_col1, rec_col2 = st.columns(2)
            with rec_col1:
                st.markdown("<h5 style='font-size: 1.1rem; margin-top: 20px; '>Immediate Actions</h5>", unsafe_allow_html=True)
                st.markdown("""
                <ul style='font-size: 1rem;'>
                    <li>15-20% retention discount</li>
                    <li>Personal call from account manager</li>
                    <li>Contract upgrade with incentives</li>
                </ul>
                """, unsafe_allow_html=True)
            with rec_col2:
                st.markdown("<h5 style='font-size: 1.1rem; margin-top: 20px; '>Follow-up</h5>", unsafe_allow_html=True)
                st.markdown("""
                <ul style='font-size: 1rem;'>
                    <li>Autopay discount offer</li>
                    <li>VIP status communication</li>
                    <li>Service bundle upgrade</li>
                </ul>
                """, unsafe_allow_html=True)

        elif churn_prob > 0.4:
            st.markdown("""
            <div style='background-color: #fff3e0; padding: 10px; border-left: 5px solid #f57c00; border-radius: 5px;'>
                <h4 style='color: #f57c00; margin-top: 0; font-size: 1.3rem;'>Medium Risk - Proactive Engagement</h4>
            </div>
            """, unsafe_allow_html=True)
            
            rec_col1, rec_col2 = st.columns(2)
            with rec_col1:
                st.markdown("<h5 style='font-size: 1.1rem; margin-top: 20px; '>Recommended Actions</h5>", unsafe_allow_html=True)
                st.markdown("""
                <ul style='font-size: 1rem;'>
                    <li>Targeted retention email</li>
                    <li>Service bundle upgrade offer</li>
                    <li>10% loyalty discount</li>
                </ul>
                """, unsafe_allow_html=True)
            with rec_col2:
                st.markdown("<h5 style='font-size: 1.1rem; margin-top: 20px; '>Monitoring</h5>", unsafe_allow_html=True)
                st.markdown("""
                <ul style='font-size: 1rem;'>
                    <li>Track usage patterns</li>
                    <li>Quarterly check-in call</li>
                    <li>Set up engagement alerts</li>
                </ul>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style='background-color: #e8f5e9; padding: 10px; border-left: 5px solid #388e3c; border-radius: 5px;'>
                <h4 style='color: #388e3c; margin-top: 0; font-size: 1.3rem;'>Low Risk - Maintain & Upsell</h4>
            </div>
            """, unsafe_allow_html=True)
            
            rec_col1, rec_col2 = st.columns(2)
            with rec_col1:
                st.markdown("<h5 style='font-size: 1.1rem; margin-top: 20px; '>Maintenance</h5>", unsafe_allow_html=True)
                st.markdown("""
                <ul style='font-size: 1rem;'>
                    <li>Customer is satisfied</li>
                    <li>Regular engagement</li>
                    <li>Referral program invite</li>
                </ul>
                """, unsafe_allow_html=True)
            with rec_col2:
                st.markdown("<h5 style='font-size: 1.1rem; margin-top: 20px; '>Growth Opportunities</h5>", unsafe_allow_html=True)
                st.markdown("""
                <ul style='font-size: 1rem;'>
                    <li>Premium service upsell</li>
                    <li>Additional line promotion</li>
                    <li>Loyalty rewards program</li>
                </ul>
                """, unsafe_allow_html=True)
            
    else:
        # Default view
        st.subheader("Why ChurnGuard ML?")
        st.markdown("""
        <p style='font-size: 1rem;'><strong>ChurnGuard ML helps telecom companies reduce customer churn by:</strong></p>
        <ul style='font-size: 1rem;'>
            <li>Identifying at-risk customers before they leave</li>
            <li>Reducing churn costs (acquiring new customers is 5-25x more expensive)</li>
            <li>Providing explainable predictions for targeted interventions</li>
            <li>Delivering actionable retention strategies</li>
        </ul>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Tabs with actual model visuals
        tab1, tab2, tab3 = st.tabs(["Model Performance", "Feature Importance", "Business Impact"])
        
        with tab1:
            st.markdown("<h3 style='font-size: 1.3rem;'>Model Validation Metrics</h3>", unsafe_allow_html=True)
            
            perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
            
            with perf_col1:
                st.metric("AUC Score", "0.8252", help="Area Under ROC Curve - Strong discrimination ability")
            
            with perf_col2:
                st.metric("Accuracy", "78%", help="Overall prediction accuracy")
            
            with perf_col3:
                st.metric("Precision", "61%", help="When we predict churn, we're right 61% of the time")
            
            with perf_col4:
                st.metric("Recall", "51%", help="We catch 51% of actual churners")
            
            st.markdown("---")
            
            # Confusion Matrix
            st.markdown("<h3 style='font-size: 1.5rem;'>Confusion Matrix</h3>", unsafe_allow_html=True)
            cm_data = {
                'Predicted No Churn': [912, 185],
                'Predicted Churn': [123, 189]
            }
            cm_df = pd.DataFrame(cm_data, index=['Actual No Churn', 'Actual Churn'])

            # Create custom hover text
            hover_text = [
                ['True Negatives: 912<br>Correctly predicted No Churn', 'False Positives: 123<br>Incorrectly predicted Churn'],
                ['False Negatives: 185<br>Missed churners', 'True Positives: 189<br>Correctly predicted Churn']
            ]

            fig_cm = px.imshow(cm_df, 
                            text_auto=True, 
                            color_continuous_scale='Blues',
                            labels=dict(x="Predicted Label", y="True Label"))
            fig_cm.update_traces(
                hovertemplate='%{customdata}<extra></extra>',
                customdata=hover_text,
                textfont_size=20
            )
            fig_cm.update_layout(
                    height=600, 
                    coloraxis_showscale=False,
                    xaxis=dict(tickfont=dict(size=14), title_font=dict(size=16)),
                    yaxis=dict(tickfont=dict(size=14), title_font=dict(size=16)),
                    font=dict(size=14)
                )
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with tab2:
            st.markdown("<h3 style='font-size: 1.3rem;'>Top 10 Most Important Features</h3>", unsafe_allow_html=True)
            
            # Feature importance data
            feature_imp = {
                'Feature': ['Contract', 'InternetService', 'OnlineSecurity', 'TechSupport', 
                        'StreamingMovies', 'tenure', 'PhoneService', 'PaperlessBilling', 
                        'MultipleLines', 'TotalCharges'],
                'Importance': [0.3868, 0.1776, 0.0755, 0.0417, 0.0302, 0.0274, 0.0241, 0.0232, 0.0218, 0.0202]
            }
            
            fig_imp = px.bar(feature_imp, x='Importance', y='Feature', orientation='h',
                            color='Importance', color_continuous_scale='Viridis')
            fig_imp.update_layout(
                    height=600, 
                    coloraxis_showscale=False,
                    xaxis=dict(tickfont=dict(size=14), title_font=dict(size=16)),
                    yaxis=dict(tickfont=dict(size=14), title_font=dict(size=16)),
                    font=dict(size=14)
                ) 
            fig_imp.update_yaxes(categoryorder='total ascending')
            st.plotly_chart(fig_imp, use_container_width=True)
            
            st.markdown("---")
            
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                st.markdown("<h5 style='font-size: 1.1rem;'>High Churn Risk Factors</h5>", unsafe_allow_html=True)
                st.markdown("""
                <ol style='font-size: 1rem;'>
                    <li>Month-to-month contracts (most important)</li>
                    <li>Electronic check payments</li>
                    <li>No online security</li>
                    <li>Fiber optic internet</li>
                    <li>Short tenure (< 6 months)</li>
                </ol>
                """, unsafe_allow_html=True)
            
            with insight_col2:
                st.markdown("<h5 style='font-size: 1.1rem;'>Retention Drivers</h5>", unsafe_allow_html=True)
                st.markdown("""
                <ol style='font-size: 1rem;'>
                    <li>Long-term contracts (1-2 years)</li>
                    <li>Automatic payment methods</li>
                    <li>Multiple services bundled</li>
                    <li>Tech support subscription</li>
                    <li>Tenure > 24 months</li>
                </ol>
                """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("<h3 style='font-size: 1.3rem;'>Business Impact Simulation</h3>", unsafe_allow_html=True)
            
            impact_col1, impact_col2 = st.columns(2)
            
            with impact_col1:
                st.markdown("<h5 style='font-size: 1.1rem;'>Churn Economics</h5>", unsafe_allow_html=True)
                st.markdown("""
                <ul style='font-size: 1rem;'>
                    <li>Average customer lifetime value: $2,400</li>
                    <li>Cost to acquire new customer: $300-500</li>
                    <li>Monthly churn rate: ~27% (industry avg)</li>
                    <li>Retention campaign cost: $50-100/customer</li>
                </ul>
                """, unsafe_allow_html=True)
            
            with impact_col2:
                st.markdown("<h5 style='font-size: 1.1rem;'>ROI Calculation</h5>", unsafe_allow_html=True)
                st.markdown("""
                <p style='font-size: 1rem;'>For 10,000 customers:</p>
                <ul style='font-size: 1rem;'>
                    <li>Churners identified: ~1,350 (27% base rate)</li>
                    <li>Prevented churn (30% success): 405 customers</li>
                    <li>Revenue saved: $972,000</li>
                    <li>Campaign cost: $135,000</li>
                    <li>Net benefit: $837,000</li>
                </ul>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background-color: #e8f5e9; padding: 20px; border-left: 5px solid #388e3c; border-radius: 5px; margin-top: 20px;'>
                <h4 style='color: #388e3c; margin-top: 0; font-size: 1.5rem;'>ROI: 520%</h4>
                <p style='margin-bottom: 0; font-size: 1rem;'>Every $1 spent on retention saves $6.20 in customer lifetime value</p>
            </div>
            """, unsafe_allow_html=True)