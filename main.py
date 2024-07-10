import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier

# Load data function
@st.cache
def load_data():
    d1 = pd.read_csv('https://raw.githubusercontent.com/Bre19/ASDTest-Python/main/data/Toddler%20Autism%20dataset%20July%202018.csv')
    d2 = pd.read_csv('https://raw.githubusercontent.com/Bre19/ASDTest-Python/main/data/autism_screening.csv')
    d3 = pd.read_csv('https://raw.githubusercontent.com/Bre19/ASDTest-Python/main/data/data_csv.csv')
    return d1, d2, d3

# Preprocess data function
@st.cache
def preprocess_data(d1, d2, d3):
    # Convert ages from months to years if column exists
    if 'Age_Mons' in d1.columns:
        d1["Age_Mons"] = (d1["Age_Mons"] / 12).astype(int)
    if 'age' in d3.columns:
        d3["age"] = (d3["age"] / 12).astype(int)
    else:
        # Handle the case where 'age' column is missing
        st.warning("Column 'age' is missing in d3")

    # Drop rows with missing values
    d2 = d2.dropna()
    d3 = d3.dropna()

    # Ensure consistency in column names
    d1 = d1.iloc[:, 1:]
    d2 = pd.concat([d2.iloc[:, 1:11], d2.iloc[:, [12, 13, 22, 23, 24, 25, 26, 27]]], axis=1)
    d3 = pd.concat([d3.iloc[:, 0:11], d3.iloc[:, [17, 11, 12, 13, 14, 19, 20]]], axis=1)

    d1.columns = d2.columns
    d3.columns = d2.columns

    # Combine datasets
    data = pd.concat([d1, d2, d3], axis=0)

    # Replacement dictionary for inconsistent categorical values
    replacements = {
        'f': 'F',
        'm': 'M',
        'yes': 'Yes',
        'no': 'No',
        'YES': 'Yes',
        'NO': 'No',
        'middle eastern': 'Middle Eastern',
        'Middle Eastern ': 'Middle Eastern',
        'mixed': 'Mixed',
        'asian': 'Asian',
        'black': 'Black',
        'south asian': 'South Asian',
        'PaciFica': 'Pacifica',
        'Pasifika': 'Pacifica',
        'Health care professional': 'Health Care Professional',
        'family member': 'Family Member',
        'Family member': 'Family Member'
    }
    data = data.replace(replacements)

    # Check if columns exist
    required_columns = ['Sex', 'Jaundice', 'Family_mem_with_ASD', 'Ethnicity', 'Who_completed_the_test', 'ASD_traits']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' is missing from the dataset")

    # Split features and target variable
    X = data.drop("ASD_traits", axis=1)
    y = data["ASD_traits"]

    # Encode categorical features
    if 'Sex' in X.columns:
        sex_encoder = LabelEncoder()
        X['Sex'] = sex_encoder.fit_transform(X['Sex'])

    if 'Jaundice' in X.columns:
        jaundice_encoder = LabelEncoder()
        X['Jaundice'] = jaundice_encoder.fit_transform(X['Jaundice'])

    if 'Family_mem_with_ASD' in X.columns:
        family_mem_with_asd_encoder = LabelEncoder()
        X['Family_mem_with_ASD'] = family_mem_with_asd_encoder.fit_transform(X['Family_mem_with_ASD'])

    # One-hot encode remaining categorical features
    categorical_features = ["Ethnicity", "Who_completed_the_test"]
    X = pd.get_dummies(X, columns=[col for col in categorical_features if col in X.columns], drop_first=True)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, sex_encoder, jaundice_encoder, family_mem_with_asd_encoder

# Load and preprocess data
d1, d2, d3 = load_data()
X_scaled, y, scaler, sex_encoder, jaundice_encoder, family_mem_with_asd_encoder = preprocess_data(d1, d2, d3)

# Split data into training and testing sets
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize and train the model
model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Streamlit app
st.title('Autism Spectrum Disorder Prediction')
st.write("## Model Performance:")
st.write(f"**Accuracy:** {accuracy:.2f}")
st.write(f"**Precision:** {precision:.2f}")
st.write(f"**Recall:** {recall:.2f}")
st.write(f"**F1 Score:** {f1:.2f}")
