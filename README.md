# Medical Cost Prediction Project

This project applies machine learning models to predict medical insurance costs using various regression models, including **Linear Regression**, **Ridge Regression**, **Random Forest**, and **XGBoost**. The dataset used in this project is based on medical charges billed by health insurance in the U.S. The goal is to predict individual medical costs based on a variety of factors like age, sex, BMI, number of children, smoking status, and region.

## Project Overview

### Models Used:
1. **Linear Regression** - Baseline model.
2. **Ridge Regression** - A variation of linear regression that introduces regularization to reduce overfitting.
3. **Random Forest** - A powerful ensemble model based on decision trees.
4. **XGBoost** - A high-performance gradient-boosting model optimized for speed and accuracy.

The models were evaluated using metrics such as:
- **Mean Squared Error (MSE)**
- **R-Squared (R²) Score**
- **Cross-Validation R² Score**

## Dataset

The dataset includes the following features:
- `age`: Age of the primary beneficiary.
- `sex`: Gender of the insurance contractor (0 for female, 1 for male).
- `bmi`: Body Mass Index.
- `children`: Number of dependents covered by the insurance.
- `smoker`: Whether the beneficiary is a smoker (1 for yes, 0 for no).
- `region`: The beneficiary's region in the U.S. (encoded with one-hot encoding).
- `charges`: Individual medical costs billed by health insurance (the target variable).

## Data Preprocessing

- **Handling Missing/Redundant Data**: No missing data was detected.
- **Label Encoding**: Categorical variables like `sex` and `smoker` were label-encoded as binary values (0 and 1).
- **One-Hot Encoding**: The `region` column was one-hot encoded into `region_northwest`, `region_southeast`, and `region_southwest`.
- **Scaling**: Continuous numerical features such as `age`, `bmi`, and `children` were standardized using `StandardScaler`.
- **Log Transformation**: The `charges` column was log-transformed to handle skewness and improve model performance.

## Model Evaluation

### Linear Regression Results:
- **Mean Squared Error (MSE)**: `0.1755`
- **R-Squared (R²)**: `0.8047`
- **Cross-Validation R²**: `0.7513`

### Ridge Regression Results:
- **Mean Squared Error (MSE)**: `0.1756`
- **R-Squared (R²)**: `0.8046`

### Random Forest Results:
- **Best Parameters**: `{'max_depth': 20, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 500}`
- **Mean Squared Error (MSE)**: `0.1307`
- **R-Squared (R²)**: `0.8546`
- **Cross-Validation R²**: `0.8116`

### XGBoost Results:
- **Best Parameters**: `{'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 500}`
- **Mean Squared Error (MSE)**: `0.1979`
- **R-Squared (R²)**: `0.7798`
- **Cross-Validation R²**: `0.7429`

## Project Structure
medical-cost-prediction/ ├── data/ # Folder containing the dataset (not included in the repo) ├── src/ # Source folder containing scripts │ ├── preprocessing.py # Data preprocessing script │ ├── train_models.py # Model training script ├── tests/ # Unit tests for the project ├── requirements.txt # List of dependencies ├── README.md # Project overview and instructions (this file) └── app.py # Flask API to serve predictions


## Installation Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/medical-cost-prediction.git
   cd medical-cost-prediction
