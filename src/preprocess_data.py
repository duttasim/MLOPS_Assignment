import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load dataset
data = pd.read_csv('../data/diabetes_prediction_dataset.csv')
# Replace 'No Info' with NaN and handle missing values
data.replace('No Info', np.nan, inplace=True)

# Fill missing values for 'smoking_history' with mode (most frequent value)
data['smoking_history'].fillna(data['smoking_history'].mode()[0], inplace=True)

# Encode categorical features
label_encoders = {}
for column in ['gender', 'smoking_history']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))
    label_encoders[column] = le

data.rename(columns={'diabetes': 'target'}, inplace=True)
pd.DataFrame(data).to_csv('../data_git/diabetes.csv', index=False)

