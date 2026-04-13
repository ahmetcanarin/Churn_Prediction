# 📉 Telco Customer Churn Prediction | End-to-End Machine Learning Project

A comprehensive machine learning project designed to predict customer churn in a telecommunications company using advanced data preprocessing, feature engineering, and model optimization techniques.

---

## 🧩 Business Problem

Customer churn is one of the most critical challenges in the telecommunications industry.

Losing customers directly impacts:

- Revenue  
- Customer lifetime value (CLTV)  
- Marketing efficiency  

The goal of this project is:

> To build a machine learning model that predicts whether a customer will churn, enabling proactive retention strategies.

---

## 🎯 Project Objectives

- Perform **exploratory data analysis (EDA)**  
- Identify churn-driving behavioral patterns  
- Apply **feature engineering based on domain knowledge**  
- Handle missing values and data inconsistencies  
- Compare multiple classification algorithms  
- Optimize the best-performing model  
- Provide a scalable churn prediction pipeline  

---

## 📊 Dataset Overview

- Dataset: Telco Customer Churn  
- Observations: 7043 customers  
- Features: 44 variables  

The dataset includes:

- Customer demographics  
- Subscription details  
- Service usage  
- Payment behavior  
- Churn label (target variable)  

---

## 🔍 Exploratory Data Analysis (EDA)

Key steps performed:

- Dataset structure analysis  
- Categorical vs numerical feature identification  
- Distribution analysis  
- Target variable analysis (Churn vs features)  
- Missing value detection  
- Outlier analysis using IQR  

### 🔎 Key Insight

> Missing values in `TotalCharges` are not random —  
> they correspond to customers with very low tenure.

---

## 🛠️ Data Preprocessing

### 🔧 Data Cleaning

- Converted `TotalCharges` from object → numeric  
- Replaced blank values with `NaN`  
- Standardized inconsistent labels:
  - `"No internet service"` → `"No"`  
  - `"No phone service"` → `"No"`  

---

### 🔄 Binary Transformation

- Converted `"Yes"/"No"` variables into binary format (1/0)

---

### ⚠️ Missing Value Handling

- `TotalCharges` missing values filled with `0`  
  → Represents customers with no accumulated billing yet  

---

## ✨ Feature Engineering

Feature engineering is one of the strongest parts of this project.

### 📊 Created Features

- **AverageCharges** → normalized spending  
- **HasInternet** → internet usage flag  
- **HasFiber** → high-speed internet indicator  
- **IsMonthToMonth** → contract risk indicator  
- **TotalServices** → number of subscribed services  
- **tenure_year** → customer lifecycle segmentation  
- **IsNewCustomer** → new user flag  
- **IsAutoPayment** → automated billing indicator  
- **IsElectronicCheck** → payment behavior signal  
- **MonthlyTotalRatio** → spending consistency  

---

### 💡 Business Insight

- Month-to-month customers show **higher churn risk**  
- Customers with fewer services are **more likely to churn**  
- Auto-payment users tend to have **lower churn probability**  

---

## 🔄 Encoding Strategy

### 🔹 Label Encoding
Applied to binary categorical variables  

### 🔹 One-Hot Encoding
Applied to multi-class categorical variables  

---

## ⚙️ Feature Scaling

- Applied **StandardScaler** to numerical variables  
- Critical for:
  - Logistic Regression  
  - KNN  
  - SVM  

---

## 🤖 Modeling

### 📌 Models Evaluated

- Logistic Regression  
- K-Nearest Neighbors  
- Decision Tree (CART)  
- Random Forest  
- Support Vector Machine (SVM)  
- XGBoost  
- LightGBM  
- CatBoost  

---

### 📊 Evaluation Metrics

Models were evaluated using:

- Accuracy  
- F1 Score  
- ROC-AUC  
- Precision  
- Recall  

> Using multiple metrics ensures balanced evaluation for imbalanced classification problems.

---

## 🏆 Final Model

### ✅ CatBoost Classifier

Selected due to:

- Strong performance across metrics  
- Robustness to categorical patterns  
- Stability against overfitting  

---

## ⚙️ Hyperparameter Optimization

GridSearchCV was applied on:

- iterations  
- learning_rate  
- depth  

This improved:

- Model generalization  
- Predictive performance  

---

## 📈 Model Evaluation

Final model evaluated using **10-fold cross-validation**:

- High ROC-AUC score  
- Balanced precision and recall  
- Strong F1 score  

---

## 🚀 Prediction Example

The model can generate predictions for new customer profiles:

```python
sample = X.sample(1)
prediction = model.predict(sample)

## 🛠️ Tech Stack

### Language
- Python

### Libraries
- pandas  
- numpy  
- scikit-learn  
- catboost  
- lightgbm  
- xgboost  

### Visualization
- matplotlib  
- seaborn  

---

## 🔍 Key Strengths of the Project

- Strong EDA + data understanding  
- Domain-driven feature engineering  
- Correct interpretation of missing values  
- Multi-model benchmarking  
- Robust evaluation strategy  
- End-to-end ML pipeline  

---

## 💡 What Makes This Project Stand Out?

This is not just a classification model.

It demonstrates:

- Thinking like a data scientist  
- Understanding customer behavior  
- Translating data into business value  

---

## 🔮 Future Improvements

- Class imbalance handling (SMOTE / class weights)  
- Feature selection optimization  
- Model stacking / ensemble  
- Deployment (API / dashboard)  
- Real-time churn prediction  
