import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st
import joblib  # For saving and loading objects
import os

# Global variables for data and model
df = None
sex_encoder = None
jaundice_encoder = None
family_mem_with_asd_encoder = None
scaler = None
model = None

# Function to load data from URL
@st.cache_data
def load_data():
    global df
    url = "https://raw.githubusercontent.com/Bre19/ASDTest-Python/main/data/Toddler%20Autism%20dataset%20July%202018.csv"
    df = pd.read_csv(url)
    return df

def preprocess_data(df):
    df.replace({"Yes": 1, "No": 0}, inplace=True)
    
    global sex_encoder, jaundice_encoder, family_mem_with_asd_encoder, scaler
    
    sex_encoder = LabelEncoder()
    jaundice_encoder = LabelEncoder()
    family_mem_with_asd_encoder = LabelEncoder()
    
    sex_encoder.fit(df["Sex"])
    jaundice_encoder.fit(df["Jaundice"])
    family_mem_with_asd_encoder.fit(df["Family_mem_with_ASD"])
    
    X = df.drop("Class/ASD Traits ", axis=1)
    y = df["Class/ASD Traits "]
    
    X["Sex"] = sex_encoder.transform(X["Sex"])
    X["Jaundice"] = jaundice_encoder.transform(X["Jaundice"])
    X["Family_mem_with_ASD"] = family_mem_with_asd_encoder.transform(X["Family_mem_with_ASD"])
    
    X = pd.get_dummies(X, columns=["Ethnicity", "Who completed the test"], drop_first=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the encoders and scaler
    joblib.dump(sex_encoder, 'sex_encoder.pkl')
    joblib.dump(jaundice_encoder, 'jaundice_encoder.pkl')
    joblib.dump(family_mem_with_asd_encoder, 'family_mem_with_asd_encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    return X_scaled, y

def build_ann(input_dim):
    model = Sequential()
    model.add(Dense(64, activation="relu", input_dim=input_dim))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_model(X_train, y_train):
    model = build_ann(X_train.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)
    
    # Save the trained model
    model.save('asd_model.h5')

# Load saved objects
def load_saved_objects():
    global sex_encoder, jaundice_encoder, family_mem_with_asd_encoder, scaler, model
    sex_encoder = joblib.load('sex_encoder.pkl')
    jaundice_encoder = joblib.load('jaundice_encoder.pkl')
    family_mem_with_asd_encoder = joblib.load('family_mem_with_asd_encoder.pkl')
    scaler = joblib.load('scaler.pkl')
    model = load_model('asd_model.h5')

def prepare_model_and_data():
    global df
    X_scaled, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    train_model(X_train, y_train)
    return X_test, y_test

# Function to make predictions
def predict_asd(input_data):
    input_data = pd.DataFrame([input_data])
    input_data.replace({"Yes": 1, "No": 0}, inplace=True)
    
    # Transform categorical features
    input_data["Sex"] = sex_encoder.transform(input_data["Sex"])
    input_data["Jaundice"] = jaundice_encoder.transform(input_data["Jaundice"])
    input_data["Family_mem_with_ASD"] = family_mem_with_asd_encoder.transform(input_data["Family_mem_with_ASD"])
    input_data = pd.get_dummies(input_data, columns=["Ethnicity", "Who completed the test"], drop_first=True)
    
    # Ensure input data has the same columns as the training data
    input_data = input_data.reindex(columns=df.columns.difference(["Class/ASD Traits "]), fill_value=0)
    
    input_data = scaler.transform(input_data)
    input_data = input_data.astype('float32')
    
    prediction = model.predict(input_data)
    return prediction[0][0]

# Streamlit user interface
st.title("ASD Screening Test")

# Load data initially
df = load_data()

# Input user for 10 questions and other features
sex = st.selectbox("Sex", ["Male", "Female"])
age_mons = st.number_input("Age (in months)", min_value=0)
jaundice = st.selectbox("Jaundice", ["Yes", "No"])
family_asd = st.selectbox("Family member with ASD", ["Yes", "No"])
ethnicity = st.selectbox("Ethnicity", df["Ethnicity"].unique() if df is not None else [])
who_completed_test = st.selectbox("Who completed the test", df["Who completed the test"].unique() if df is not None else [])

# Collect answers to 10 questions
questions = {
    "A1": st.selectbox("Does your child look at you when you call his/her name?", ["Yes", "No"]),
    "A2": st.selectbox("How easy is it for you to get eye contact with your child?", ["Yes", "No"]),
    "A3": st.selectbox("Does your child point to indicate that s/he wants something? (e.g. a toy that is out of reach)", ["Yes", "No"]),
    "A4": st.selectbox("Does your child point to share interest with you? (e.g. pointing at an interesting sight)", ["Yes", "No"]),
    "A5": st.selectbox("Does your child pretend? (e.g. care for dolls, talk on a toy phone)", ["Yes", "No"]),
    "A6": st.selectbox("Does your child follow where you’re looking?", ["Yes", "No"]),
    "A7": st.selectbox("If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them? (e.g. stroking hair, hugging them)", ["Yes", "No"]),
    "A8": st.selectbox("Would you describe your child’s first words as:", ["Yes", "No"]),
    "A9": st.selectbox("Does your child use simple gestures? (e.g. wave goodbye)", ["Yes", "No"]),
    "A10": st.selectbox("Does your child stare at nothing with no apparent purpose?", ["Yes", "No"]),
}

input_data = {
    "Sex": sex,
    "Age_Mons": age_mons,
    "Jaundice": jaundice,
    "Family_mem_with_ASD": family_asd,
    "Ethnicity": ethnicity,
    "Who completed the test": who_completed_test
}

for key, answer in questions.items():
    input_data[key] = answer

# Show prediction results only when 'Submit' is clicked
if st.button("Submit"):
    # Initialize model only if it does not exist
    if not os.path.exists('asd_model.h5'):
        X_test, y_test = prepare_model_and_data()
    
    load_saved_objects()

    # Compute test accuracy only on 'Submit'
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.write(f"Test Accuracy: {test_accuracy:.2f}")
    
    prediction = predict_asd(input_data)
    if prediction > 0.5:
        st.write("The model predicts: Likely to have ASD.")
    else:
        st.write("The model predicts: Unlikely to have ASD.")
