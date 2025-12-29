# Traffic Accident Severity Prediction

## Description
This project aims to predict the severity of traffic accidents using supervised machine learning techniques. The model is trained on real-world US traffic accident data collected between 2019 and 2023. The problem is formulated as a multi-class classification task, with particular focus on handling class imbalance.

---

## Objective
- Predict traffic accident severity levels.
- Handle class imbalance effectively.
- Compare multiple supervised learning models and evaluate their performance.

---

## Dataset
- Source: US Traffic Accident Dataset  
- Time Period: 2019–2023  
- Size: Approximately 5 million records  
- Target Variable: Accident Severity  

---

## Methodology

### Data Preprocessing
- Selected relevant features and filtered the dataset by timeframe.
- Handled missing values using:
  - Median for numerical features
  - Mode and forward/backward fill for categorical and temporal features
- Optimized data types for memory efficiency.

### Exploratory Data Analysis
- Analyzed data distributions and severity imbalance.
- Visualized accident-prone states and feature correlations.

### Feature Engineering
- Created temporal features such as hour, day, month, weekday, weekend, and rush hour indicators.
- Added geographical features like state and street type.
- Grouped weather conditions and identified daytime indicators.

### Feature Selection
- Applied ANOVA F-test to select the most significant features.

### Handling Class Imbalance
- Used stratified downsampling to balance classes in the training dataset.
- Applied balanced class weights during model training.

### Model Training
- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- XGBoost  

---

## Evaluation Metrics
- Accuracy  
- Weighted F1-score  
- Macro F1-score  
- Weighted ROC-AUC  
- Confusion Matrix  

---

## Results
Gradient Boosting achieved the best overall performance, showing strong predictive capability on an imbalanced multi-class dataset.

---

## Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Imbalanced-learn  
- Matplotlib  
- Seaborn  

---

