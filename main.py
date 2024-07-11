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
    data = pd.read_csv("ASDTest-Python/data/Toddler Autism dataset July 2018.csv")
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
    
    # One-hot encoding for Ethnicity and Who_completed_the_test
    df = pd.get_dummies(df, columns=["Ethnicity", "Who completed the test"], drop_first=True)
    
    X = df.drop("Class/ASD Traits", axis=1)
    y = df["Class/ASD Traits"].apply(lambda x: 1 if x == "YES" else 0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, sex_encoder, jaundice_encoder, family_mem_with_asd_encoder

# Fungsi untuk membangun model ANN
def build_ann(input_dim):
    model = Sequential()
    model.add(Dense(64, activation="relu", input_dim=input_dim))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Load data
df = load_data()

# Preprocess data
X_scaled, y, scaler, sex_encoder, jaundice_encoder, family_mem_with_asd_encoder = preprocess_data(df)

# Split data
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build and train the model
model = build_ann(X_train_scaled.shape[1])
callback = EarlyStopping(monitor="val_loss", patience=10, verbose=1, restore_best_weights=True)
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=20, validation_split=0.2, callbacks=[callback])

# Predict
y_prob = model.predict(X_test_scaled)
y_pred = np.where(y_prob > 0.5, 1, 0)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Streamlit UI
st.title("ASD Screening Test")
st.write(f"Model Accuracy: {accuracy:.2f}")
st.write(f"Model Precision: {precision:.2f}")
st.write(f"Model Recall: {recall:.2f}")
st.write(f"Model F1 Score: {f1:.2f}")

# Form for user input
st.header("Input Data")
A1 = st.selectbox("Does your child look at you when you call his/her name?", [0, 1])
A2 = st.selectbox("How easy is it for you to get eye contact with your child?", [0, 1])
A3 = st.selectbox("Does your child point to indicate that s/he wants something? (e.g. a toy that is out of reach)", [0, 1])
A4 = st.selectbox("Does your child point to share interest with you? (e.g. pointing at an interesting sight)", [0, 1])
A5 = st.selectbox("Does your child pretend? (e.g. care for dolls, talk on a toy phone)", [0, 1])
A6 = st.selectbox("Does your child follow where you’re looking?", [0, 1])
A7 = st.selectbox("If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them? (e.g. stroking hair, hugging them)", [0, 1])
A8 = st.selectbox("Would you describe your child’s first words as:", [0, 1])
A9 = st.selectbox("Does your child use simple gestures? (e.g. wave goodbye)", [0, 1])
A10 = st.selectbox("Does your child stare at nothing with no apparent purpose?", [0, 1])
Age_Mons = st.number_input("Age in months", min_value=0, max_value=240)
Sex = st.selectbox("Sex", ["M", "F"])
Ethnicity = st.selectbox("Ethnicity", df["Ethnicity"].unique())
Jaundice = st.selectbox("Has the child been diagnosed with Jaundice?", ["Yes", "No"])
Family_mem_with_ASD = st.selectbox("Any family member with ASD?", ["Yes", "No"])
Who_completed_the_test = st.selectbox("Who completed the test?", df["Who completed the test"].unique())

# Predict button
if st.button("Predict"):
    user_data = pd.DataFrame({
        "A1": [A1], "A2": [A2], "A3": [A3], "A4": [A4], "A5": [A5], "A6": [A6],
        "A7": [A7], "A8": [A8], "A9": [A9], "A10": [A10], "Age_Mons": [Age_Mons],
        "Sex": [Sex], "Ethnicity": [Ethnicity], "Jaundice": [Jaundice],
        "Family_mem_with_ASD": [Family_mem_with_ASD], "Who completed the test": [Who_completed_the_test]
    })
    
    user_data["Sex"] = sex_encoder.transform(user_data["Sex"])
    user_data["Jaundice"] = jaundice_encoder.transform(user_data["Jaundice"])
    user_data["Family_mem_with_ASD"] = family_mem_with_asd_encoder.transform(user_data["Family_mem_with_ASD"])
    user_data = pd.get_dummies(user_data, columns=["Ethnicity", "Who completed the test"], drop_first=True)
    user_data = user_data.reindex(columns=X.columns, fill_value=0)
    
    user_data_scaled = scaler.transform(user_data)
    user_prob = model.predict(user_data_scaled)
    user_pred = np.where(user_prob > 0.5, "Yes", "No")
    
    st.write(f"Predicted ASD Traits: {user_pred[0]}")
