# 🚀 FinGuard AI – Corporate Fraud Detection & Risk Intelligence Platform

--> An Explainable AI-powered Corporate Fraud Detection System that combines advanced ensemble machine learning techniques with interpretable AI to identify fraudulent companies and support data-driven auditing decisions.


## 📖 Overview

Corporate fraud remains one of the biggest challenges in financial auditing and risk management. Traditional fraud detection methods are often time-consuming, rule-based, and unable to adapt to evolving fraudulent patterns.

As part of our research on **Explainable AI for Corporate Fraud Detection**, we developed **FinGuard AI**, an intelligent fraud risk assessment platform that leverages machine learning and explainable AI techniques to detect fraudulent companies based on financial indicators.

The platform not only predicts fraud risk with high accuracy but also explains *why* a company has been flagged, making the system suitable for auditors, financial analysts, compliance teams, and regulatory bodies.


## 🔬 Research Contribution

This project is based on research focused on improving fraud detection performance while maintaining transparency and interpretability.

### Research Objectives

* Detect fraudulent companies using financial statement data.
* Address class imbalance issues in fraud datasets.
* Improve prediction accuracy through ensemble learning.
* Provide explainable predictions using XAI techniques.
* Generate auditor-friendly reports for decision-making.

### Proposed Framework

The research introduces a **Hybrid Stacking Ensemble Model** consisting of:

* Random Forest (Base Learner)
* Gradient Boosting (Base Learner)
* Logistic Regression (Meta Learner)

To overcome the imbalance between fraudulent and non-fraudulent records:

* SMOTE (Synthetic Minority Oversampling Technique) is applied.

For model interpretability:

* SHAP (Global & Local Feature Importance)
* LIME (Instance-Level Explanations)


## ✨ Key Features

### 🔐 Secure Authentication

* User login system
* Role-based access support

### 📊 Data Analytics Dashboard

* Dataset overview
* Statistical analysis
* Missing value detection
* Feature exploration

### 🤖 Machine Learning Models

* Random Forest
* Gradient Boosting
* Stacking Ensemble (Proposed Model)

### ⚖️ Imbalanced Data Handling

* SMOTE Oversampling

### 📈 Model Evaluation

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* ROC Curve
* Precision-Recall Curve
* Calibration Curve

### 🧠 Explainable AI (XAI)

* SHAP Beeswarm Plot
* SHAP Feature Importance
* SHAP Waterfall Analysis
* SHAP Dependence Plot
* LIME Explanations
* Partial Dependence Plot (PDP)

### 🎯 Fraud Prediction Engine

* Real-time fraud prediction
* Risk probability score
* Confidence interval estimation
* Model consensus comparison

### 🔮 Fraud Risk Forecasting

* Future fraud probability estimation
* Trend analysis visualization

### 📄 Automated PDF Reporting

* Fraud Analysis Reports
* Executive Audit Reports
* Risk Classification Reports

Streamlit Dashboard


## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-Learn
* Random Forest
* Gradient Boosting
* Stacking Classifier

### Explainable AI

* SHAP
* LIME

### Data Processing

* Pandas
* NumPy

### Visualization

* Plotly
* Matplotlib
* Seaborn

### Web Framework

* Streamlit

### Data Balancing

* SMOTE (Imbalanced-Learn)

### Reporting

* ReportLab

### Model Persistence

* Joblib

## ⚙️ Installation

## 1️⃣ Clone Repository

git clone https://github.com/yourusername/FinGuard-AI.git
cd FinGuard-AI


## 2️⃣ Create Virtual Environment

## Windows : 
   python -m venv venv
   venv\Scripts\activate

## Linux / Mac  :  source venv/bin/activate


## 3️⃣ Install Dependencies

pip install streamlit
pip install pandas numpy
pip install scikit-learn
pip install imbalanced-learn
pip install shap
pip install lime
pip install matplotlib seaborn
pip install plotly
pip install reportlab
pip install joblib

## ▶️ Running the Application

Start the Streamlit server: streamlit run app.py

The application will open automatically in your browser:   http://localhost:8501



## 📋 How to Use

## Step 1

Launch the application.

## Step 2

Login using a valid account.

Example:

Username: admin / analyst
Password: admin123 / analyst123

## Step 3

Upload your corporate financial dataset (.csv).

## Step 4

Explore:

* Dataset Analysis
* Model Performance
* Explainable AI Dashboard

## Step 5

Navigate to **Prediction Module**.

## Step 6

Enter company financial indicators.

## Step 7

Run fraud prediction.

## Step 8

Review:

* Fraud Probability
* Risk Level
* SHAP Explanations
* Model Consensus
* Forecast Analysis

## Step 9

Download the generated PDF audit report.


## 📊 Model Evaluation Metrics

The system evaluates fraud detection performance using:

✔ Accuracy
✔ Precision
✔ Recall
✔ F1 Score
✔ ROC-AUC
✔ Precision-Recall AUC
✔ Calibration Score
✔ Confusion Matrix


## 🎯 Research Outcomes

* Improved fraud detection accuracy through ensemble learning.
* Reduced bias caused by imbalanced fraud datasets using SMOTE.
* Enhanced transparency with SHAP and LIME explainability.
* Delivered auditor-friendly reports for financial investigations.
* Built a production-style fraud risk intelligence platform using Streamlit.


## 📸 Application Modules

* Login & Authentication
* Dashboard Overview
* Data Analysis
* Model Comparison
* Performance Evaluation
* Explainable AI
* Fraud Prediction
* Risk Forecasting
* PDF Report Generation


## 🚀 Future Enhancements

* Deep Learning-Based Fraud Detection
* Real-Time Financial Data Integration
* API Deployment
* Cloud Hosting (AWS/Azure)
* Multi-Factor Fraud Scoring
* Blockchain-Based Audit Verification
* Automated Compliance Monitoring

## Research Areas

* Explainable AI (XAI)
* Fraud Analytics
* Financial Risk Intelligence
* Machine Learning
* Predictive Modeling

## 🔗 Project Links

https://finguard-ai-corporate-fraud-detection-risk-intelligence-platfo.streamlit.app/


## 👨‍💻 Author

**Esha Sagar S**


## 📜 License

This project is developed for academic research and educational purposes. Feel free to use, modify, and extend the project with proper attribution.

⭐ If you found this project useful, consider giving it a star on GitHub!
