import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Bre19/ASDTest-Python/main/data/Toddler%20Autism%20dataset%20July%202018.csv"
    data = pd.read_csv(url)
    return data

# Fungsi untuk melakukan preprocessing data
@st.cache_data
def preprocess_data(df):
    df = df.dropna()
    df["Age_Mons"] = (df["Age_Mons"] / 12).astype(int)
    
    # Encoding categorical variables
    sex_encoder = LabelEncoder()
    jaundice_encoder = LabelEncoder()
    family_mem_with_asd_encoder = LabelEncoder()
    
    df["Sex"] = sex_encoder.fit_transform(df["Sex"])
    df["Jaundice"] = jaundice_encoder.fit_transform(df["Jaundice"])
    df["Family_mem_with_ASD"] = family_mem_with_asd_encoder.fit_transform(df["Family_mem_with_ASD"])
    
    # One-hot encoding for Ethnicity and Who completed the test
    df = pd.get_dummies(df, columns=["Ethnicity", "Who completed the test"], drop_first=True)
    
    X = df.drop("Class/ASD Traits ", axis=1)  # Pastikan nama kolom sesuai
    y = df["Class/ASD Traits "].apply(lambda x: 1 if x == "YES" else 0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, sex_encoder, jaundice_encoder, family_mem_with_asd_encoder, X.columns

# Fungsi untuk membangun model ANN
def build_ann(input_dim):
    model = Sequential()
    model.add(Dense(64, activation="relu", input_dim=input_dim))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Main
st.title("ASD Screening Test")

df = load_data()
X_scaled, y, scaler, sex_encoder, jaundice_encoder, family_mem_with_asd_encoder, feature_columns = preprocess_data(df)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build and train the model
model = build_ann(X_train.shape[1])
callback = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=20, validation_split=0.2, callbacks=[callback])

# Predict and evaluate
y_prob = model.predict(X_test)
y_pred = (y_prob > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.write(f"Accuracy: {accuracy}")
st.write(f"Precision: {precision}")
st.write(f"Recall: {recall}")
st.write(f"F1 Score: {f1}")

# Input form for new predictions
st.header("Predict ASD")
with st.form(key="input_form"):
    A1 = st.selectbox("Does your child look at you when you call his/her name?", ["Yes", "No"])
    A2 = st.selectbox("How easy is it for you to get eye contact with your child?", ["Yes", "No"])
    A3 = st.selectbox("Does your child point to indicate that s/he wants something?", ["Yes", "No"])
    A4 = st.selectbox("Does your child point to share interest with you?", ["Yes", "No"])
    A5 = st.selectbox("Does your child pretend?", ["Yes", "No"])
    A6 = st.selectbox("Does your child follow where you’re looking?", ["Yes", "No"])
    A7 = st.selectbox("If someone in the family is upset, does your child comfort them?", ["Yes", "No"])
    A8 = st.selectbox("Would you describe your child’s first words as?", ["Yes", "No"])
    A9 = st.selectbox("Does your child use simple gestures?", ["Yes", "No"])
    A10 = st.selectbox("Does your child stare at nothing with no apparent purpose?", ["Yes", "No"])
    Age_Mons = st.number_input("Child's age in months", min_value=0, max_value=120)
    Sex = st.selectbox("Sex", ["Male", "Female"])
    Ethnicity = st.selectbox("Ethnicity", df["Ethnicity"].unique())
    Jaundice = st.selectbox("Has the child had jaundice?", ["Yes", "No"])
    Family_mem_with_ASD = st.selectbox("Any family member with ASD?", ["Yes", "No"])
    Who_completed_the_test = st.selectbox("Who completed the test?", df["Who completed the test"].unique())
    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    input_data = pd.DataFrame(
        [[A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, Age_Mons, Sex, Ethnicity, Jaundice, Family_mem_with_ASD, Who_completed_the_test]],
        columns=["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "Age_Mons", "Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD", "Who completed the test"]
    )

    # Map Yes/No to 1/0 for A1 to A10
    input_data.replace({"Yes": 1, "No": 0}, inplace=True)

    # Encode input data
    input_data["Sex"] = sex_encoder.transform(input_data["Sex"])
    input_data["Jaundice"] = jaundice_encoder.transform(input_data["Jaundice"])
    input_data["Family_mem_with_ASD"] = family_mem_with_asd_encoder.transform(input_data["Family_mem_with_ASD"])
    input_data = pd.get_dummies(input_data, columns=["Ethnicity", "Who completed the test"], drop_first=True)

    # Handle missing columns due to one-hot encoding
    missing_cols = set(feature_columns) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0

    input_data = input_data[feature_columns]

    # Scale input data
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction_prob = model.predict(input_data_scaled)
    prediction = (prediction_prob > 0.5).astype(int)

    result = "Positive for ASD" if prediction[0] == 1 else "Negative for ASD"
    st.write(f"Prediction: {result}")
