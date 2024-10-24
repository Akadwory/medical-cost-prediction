import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Strip column names
    df.columns = df.columns.str.strip()
    
    # Encode 'sex' and 'smoker'
    df['sex'] = df['sex'].map({'female': 0, 'male': 1})
    df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
    
    # One-hot encode 'region'
    df = pd.get_dummies(df, columns=['region'], drop_first=True)
    
    # Log-transform 'charges'
    df['log_charges'] = np.log(df['charges'] + 1)
    
    # Standardize numerical columns
    scaler = StandardScaler()
    df[['age', 'bmi', 'children']] = scaler.fit_transform(df[['age', 'bmi', 'children']])
    
    return df
