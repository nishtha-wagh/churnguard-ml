# ChurnGuard ML

**Customer Churn Prediction System** - An end-to-end machine learning solution for predicting telecom customer churn using XGBoost and SHAP explainability.

![Python](https://img.shields.io/badge/Python-3.13-blue)  ![Streamlit](https://img.shields.io/badge/Streamlit-1.54-red)

## Overview

ChurnGuard ML helps telecom companies reduce customer churn by:
- Identifying at-risk customers before they leave
- Reducing churn costs (acquiring new customers is 5-25x more expensive)
- Providing explainable predictions for targeted interventions
- Delivering actionable retention strategies with ROI calculations

## Live Demo

**[Try it here!](https://churnguard-ml.streamlit.app)** *(update after deployment)*

## Model Performance

- **AUC Score:** 0.8252
- **Accuracy:** 78%
- **Precision (Churn):** 61%
- **Recall (Churn):** 51%

## Tech Stack

- **ML Framework:** XGBoost
- **Explainability:** SHAP
- **Backend:** Python, Pandas, NumPy, scikit-learn
- **Frontend:** Streamlit
- **Deployment:** Streamlit Community Cloud
- **Data:** Telco Customer Churn Dataset (7,043 customers)

## Project Structure
```
churnguard-ml/
├── data/                      # Dataset
├── notebooks/                 # Jupyter notebooks for EDA and modeling
│   └── 01_eda_and_modeling.ipynb
├── models/                    # Trained models and artifacts
│   ├── churnguard_model.joblib
│   ├── label_encoders.joblib
│   └── feature_names.joblib
├── app/                       # Streamlit application
│   ├── streamlit_app.py       # Main app
│   └── main.py                # FastAPI (optional)
├── requirements.txt
└── README.md
```

## Key Features

### 1. Predictive Model
- **XGBoost classifier** trained on 7,043 telecom customers
- **Feature engineering:** Customer value, tenure groups, service bundles
- **Top predictive features:** Contract type, Internet service, Online security

### 2. Interactive Dashboard
- Real-time churn risk prediction
- Visual risk gauge with color-coded alerts
- Customizable customer inputs
- Business impact simulation with ROI calculations

### 3. Explainable AI
- SHAP values for model interpretability
- Feature importance visualization
- Confusion matrix and performance metrics
- Actionable retention recommendations

## Local Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/churnguard-ml.git
cd churnguard-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook (to train model)
jupyter notebook notebooks/01_eda_and_modeling.ipynb

# Run Streamlit app
streamlit run app/streamlit_app.py
```

## Business Impact

**ROI Simulation (10,000 customers):**
- Churners identified: ~1,350 (27% base rate)
- Prevented churn (30% success): 405 customers
- Revenue saved: $972,000
- Campaign cost: $135,000
- **Net benefit: $837,000**
- **ROI: 520%** - Every $1 spent saves $6.20 in customer lifetime value

## Model Training

The model was trained using:
- **Algorithm:** XGBoost with hyperparameter tuning
- **Features:** 19 customer attributes (demographics, services, billing)
- **Target:** Binary churn (Yes/No)
- **Validation:** Train/test split (80/20) with stratification
- **Metrics:** AUC-ROC, Precision, Recall, F1-Score

Top 3 features by importance:
1. Contract type (0.387)
2. Internet service (0.178)
3. Online security (0.076)

## Use Cases

- **Retention teams:** Identify high-risk customers for proactive outreach
- **Product teams:** Understand service features that drive retention
- **Finance teams:** Calculate ROI of retention campaigns
- **Marketing teams:** Target at-risk segments with personalized offers

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## License

MIT License - feel free to use this project for learning or commercial purposes.

## Author

**Nishtha Wagh**
- Portfolio: [TheDataSkeptic](https://thedataskeptic.vercel.app/)
- LinkedIn: [linkedin.com/in/nishthawagh](https://linkedin.com/in/nishthawagh)
- GitHub: [@Nishtha Wagh](https://github.com/nishtha-wagh)

## Acknowledgments

- Dataset: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Libraries: XGBoost, SHAP, Streamlit, Plotly