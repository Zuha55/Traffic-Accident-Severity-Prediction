Project Summary: Traffic Accident Severity Prediction
Objective
Developed a supervised machine learning solution to predict traffic accident severity using real-world US accident data (2019-2023). The project addresses a multi-class classification problem with a strong focus on handling class imbalance.

Key Contributions & Technical Skills Demonstrated
Data Acquisition & Preprocessing: Managed a large-scale dataset (~5 million records), loading and filtering for relevant timeframes and features. Implemented robust missing value imputation (median for numeric, mode/FFill/BFill for categorical/temporal) and optimized data types for memory efficiency.

Exploratory Data Analysis (EDA): Performed comprehensive analysis to understand data distributions, identify class imbalance issues (e.g., Severity 2 dominating ~83% of records), and visualize key insights like top accident-prone states and feature correlations.

Feature Engineering: Created impactful features from raw data, including temporal aspects (hour, day, month, weekday, rush hour, weekend flags), geographical indicators (street type, state), and environmental conditions (grouped weather, daytime indicator). This enhanced model understanding of complex patterns.

Feature Selection: Applied ANOVA F-test to identify and select the top 15 most statistically significant features influencing accident severity, reducing dimensionality and improving model focus.

Class Imbalance Handling: Addressed severe class imbalance through stratified downsampling of the training data (balancing classes to 50,000 samples each) and computing balanced class weights for model training. This ensures models learn effectively from minority classes.

Model Development & Evaluation: Implemented and trained a suite of supervised machine learning models, including Logistic Regression, Random Forest, Gradient Boosting, and XGBoost. Critically evaluated performance using appropriate metrics for imbalanced multi-class problems: Weighted F1-score (primary), Weighted ROC-AUC, Accuracy, and Macro F1-score. Utilized confusion matrices and ROC curves for in-depth performance analysis.

Key Findings: Demonstrated that Gradient Boosting achieved the best overall performance with a Weighted F1-score of ~0.62 and strong discriminative power (Weighted ROC-AUC of ~0.83), highlighting its effectiveness in handling the complexity and imbalance of the dataset.

Technologies Used
Programming Languages: Python
Libraries: Pandas, NumPy, Scikit-learn (StandardScaler, LabelEncoder, train_test_split, f_classif, classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, roc_curve), XGBoost, Imbalanced-learn (SMOTE - though downsampling was used for final models), Matplotlib, Seaborn.
