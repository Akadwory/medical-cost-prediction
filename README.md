# Medical Cost Prediction Project

This project applies machine learning models to predict medical insurance costs using various regression models, including **Linear Regression**, **Ridge Regression**, **Random Forest**, and **XGBoost**. The dataset used in this project is based on medical charges billed by health insurance in the U.S. The goal is to predict individual medical costs based on a variety of factors like age, sex, BMI, number of children, smoking status, and region.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Source](#dataset-source)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Evaluation](#model-evaluation)
5. [Installation Instructions](#installation-instructions)
6. [Running the Flask API](#running-the-flask-api)
7. [Usage and Reproduction](#usage-and-reproduction)
8. [Next Steps for Improvement](#next-steps-for-improvement)
9. [Contributions](#contributions)
10. [License](#license)

---

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

---

## Dataset Source

The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance). It contains data on medical insurance costs for individuals based on several features.

- **Title**: Insurance Medical Charges Dataset
- **Source**: [Insurance Dataset on Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **License**: This dataset is publicly available on Kaggle.

You can download the dataset directly from Kaggle. To use it in this project:
1. Download the dataset from the provided link.
2. Place the downloaded CSV file in the `data/` directory in the root of the project.

**Features in the dataset**:
- `age`: Age of the primary beneficiary.
- `sex`: Gender of the insurance contractor (0 for female, 1 for male).
- `bmi`: Body Mass Index.
- `children`: Number of dependents covered by the insurance.
- `smoker`: Whether the beneficiary is a smoker (1 for yes, 0 for no).
- `region`: The beneficiary's region in the U.S. (encoded with one-hot encoding).
- `charges`: Individual medical costs billed by health insurance (the target variable).

**Acknowledgment**:
We gratefully acknowledge the dataset's creator, [Miri Choi](https://www.kaggle.com/mirichoi0218), and the Kaggle platform for making this dataset publicly available.

---

## Data Preprocessing

- **Handling Missing/Redundant Data**: No missing data was detected.
- **Label Encoding**: Categorical variables like `sex` and `smoker` were label-encoded as binary values (0 and 1).
- **One-Hot Encoding**: The `region` column was one-hot encoded into `region_northwest`, `region_southeast`, and `region_southwest`.
- **Scaling**: Continuous numerical features such as `age`, `bmi`, and `children` were standardized using `StandardScaler`.
- **Log Transformation**: The `charges` column was log-transformed to handle skewness and improve model performance.

---

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

---

## Installation Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/medical-cost-prediction.git
   cd medical-cost-prediction

2. **Create and activate a virtual environment:**
python -m venv myenv
source myenv/bin/activate  # On macOS/Linux
myenv\Scripts\activate     # On Windows

3. **Install dependencies:**
pip install -r requirements.txt

**Download and place the dataset:**

Download the dataset and place it in the data/ directory.

**Run the Jupyter notebook:**

If you'd like to explore the data analysis and model training interactively:
jupyter notebook


**Running the Flask API**
1. **Run the Flask application:**
   python app.py
   The API will be hosted at http://127.0.0.1:5001.
3. **Make predictions:**
You can send POST requests to the /predict endpoint with the necessary features.
curl -X POST http://127.0.0.1:5001/predict -H "Content-Type: application/json" -d '{"features": [29, 1, 25.3, 0, 0, 0, 1, 0]}'
This will return a JSON response with the predicted medical cost.


**Usage and Reproduction**
You can modify and improve the models by changing the hyperparameters in src/train_models.py.
Additional unit tests can be added to the tests/ folder to ensure reproducibility and accuracy of the code.
Use the requirements.txt file to replicate the development environment.

**Next Steps for Improvement**
Based on the current results, the following steps could further improve the models:

**Feature Engineering:** Additional interaction terms or polynomial features could be explored to capture non-linear relationships.
Hyperparameter Tuning: Further tuning of hyperparameters for the Random Forest and XGBoost models.
Advanced Ensemble Models: Implement more advanced ensemble techniques like stacking different models to achieve better accuracy.
Contributions
Feel free to fork this project and contribute! If you find any issues or improvements, open an issue or a pull request.
