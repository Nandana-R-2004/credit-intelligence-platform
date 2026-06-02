# 🏦 Credit Intelligence Platform

## 📌 Overview

The **Credit Intelligence Platform** is a machine learning-based web application that evaluates the credit risk of loan applicants and predicts the probability of loan default.

It analyzes financial, behavioral, and credit-related data to generate a **risk score and classification**, helping financial institutions make data-driven lending decisions.

The platform is deployed using an interactive **Streamlit dashboard** and supports real-time predictions.

---

## ✨ Key Features

- 🔍 Credit Risk Prediction using Machine Learning  
- 📊 Default Probability Estimation  
- 🏷️ Risk Categorization (Low / Medium / High)  
- 🌐 Interactive Streamlit Web Interface  
- ⚡ Real-time Prediction System  
- 🧠 Financial Decision Support System  
- 🐳 Dockerized Deployment Support  

---

## 🎯 Problem Statement

Financial institutions need a reliable system to assess loan default risk before approving credit applications.

This system predicts credit risk based on:

- Applicant financial stability  
- Credit history behavior  
- Income level  
- Loan amount requested  
- Employment details  

It provides a probabilistic risk score to support lending decisions.

---

## 🤖 Machine Learning Model

### Algorithm Used
- XGBoost Classifier  

---

### 📥 Input Features

| Feature | Description |
|--------|-------------|
| Age | Applicant age |
| Work Experience | Years of employment |
| Annual Income | Yearly income |
| Loan Amount Requested | Credit amount applied for |
| Credit History Score | Past credit behavior |
| Financial Stability Score | Financial strength indicator |
| External Credit Rating | External credit evaluation |

---

### 🎯 Output

- Default Probability  
- Risk Category (Low / Medium / High)  
- Lending Recommendation  

---

## ⚖️ Risk Classification

| Default Probability | Risk Category | Decision |
|---------------------|--------------|----------|
| < 35% | Low Risk | Approved |
| 35% – 65% | Medium Risk | Manual Review Required |
| > 65% | High Risk | Special Review Required |

---

## 🌍 Financial Inclusion Approach

This system ensures **responsible lending practices** by not rejecting applicants solely based on risk score.

Instead, high-risk cases are:
- Reviewed manually  
- Evaluated for additional financial context  
- Considered for collateral or guarantor-based approval  

---

## 🛠️ Tech Stack

**Programming Language:** Python  
**Machine Learning:** XGBoost, Scikit-Learn, Pandas, NumPy  
**Frontend:** Streamlit  
**Deployment:** Docker  

---

## 📁 Project Structure
CREDIT_RISK_PLATFORM/
│
├── data/
├── documents/
├── models/
├── notebooks/
├── sql/
├── src/
│ ├── ml/
│ ├── ui/
│ ├── explainability/
│ ├── rules/
│ ├── utils/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md


---

## 🚀 Next Steps

- Add model performance metrics (AUC, F1-score)  
- Deploy Streamlit app on cloud (Render / AWS / HuggingFace)  
- Add SHAP explainability visuals  
- Improve UI with charts and graphs  
- Add authentication system  

---