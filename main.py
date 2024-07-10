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
  d1 = pd.read_csv('ASDTest-Python/data/Toddler Autism dataset July 2018.csv')
  d2 = pd.read_csv('ASDTest-Python/data/autism_screening.csv')
  d3 = pd.read_csv('ASDTest-Python/data/data_csv.csv')
  return d1, d2, d3

# Preprocess data function
@st.cache
def preprocess_data(d1, d2, d3):
  d1["Age_Mons"] = (d1["Age_Mons"] / 12).astype(int)
  d2 = d2.dropna()
  d3 = d3.dropna()
  d3["age"] = (d3["age"] / 12).astype(int)

  d1 = d1.iloc[:, 1:]
  d2 = pd.concat([d2.iloc[:, 1:11], d2.iloc[:, [12, 13, 22, 23, 24, 25, 26, 27]]], axis=1)
  d3 = pd.concat([d3.iloc[:, 0:11], d3.iloc[:, [17, 11, 12, 13, 14, 19, 20]]], axis=1)

  d1.columns = d2.columns
  d3.columns = d2.columns

  data = pd.concat([d1, d2, d3], axis=0)

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

  X = data.drop("ASD_traits", axis=1)
  y = data["ASD_traits"]

  sex_encoder = LabelEncoder()
  sex_encoder.fit(X['Sex'])
  X['Sex'] = sex_encoder.transform(X['Sex'])

  jaundice_encoder = LabelEncoder()
  jaundice_encoder.fit(X['Jaundice'])
  X['Jaundice'] = jaundice_encoder.transform(X['Jaundice'])

  family_mem_with_asd_encoder = LabelEncoder()
  family_mem_with_asd_encoder.fit(X['Family_mem_with_ASD'])
  X['Family_mem_with_ASD'] = family_mem_with_asd_encoder.transform(X['Family_mem_with_ASD'])

  X = pd.get_dummies(X, columns=["Ethnicity", "Who_completed_the_test"], drop_first=True)
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  return X_scaled, y, scaler, sex_encoder, jaundice_encoder, family_mem_with_asd_encoder

# Load and preprocess data
d1, d2, d3 = load_data()
X_scaled, y, scaler, sex_encoder, jaundice_encoder, family_mem_with_asd_encoder = preprocess_data(d1, d2, d3)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y,
