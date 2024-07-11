import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model  # pastikan load_model diimpor
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st
import joblib
import os

# Global variables for data and model
df = None
sex_encoder = None
jaundice_encoder = None
family_mem_with_asd_encoder = None
scaler = None
model = None
X_test = None
y_test = None
feature_columns = None  # To store feature column names

# Function to load data from URL
@st.cache_data
def load_data():
    global df
    url = "https://raw.githubusercontent.com/Bre19/ASDTest-Python/main/data/Toddler%20Autism%20dataset%20July%202018.csv"
    df = pd.read_csv(url)
    return df

def preprocess_data(df):
    df.replace({"Yes": 1, "No": 0}, inplace=True)
    
    global sex_encoder, jaundice_encoder, family_mem_with_asd_encoder, scaler, feature_columns
    
    # Fit label encoders
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
    
    feature_columns = X.columns.tolist()  # Save feature columns for later
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the encoders, scaler, and feature columns
    joblib.dump(sex_encoder, 'sex_encoder.pkl')
    joblib.dump(jaundice_encoder, 'jaundice_encoder.pkl')
    joblib.dump(family_mem_with_asd_encoder, 'family_mem_with_asd_encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(feature_columns, 'feature_columns.pkl')
    
    return X_scaled, y

def build_ann(input_dim):
    model = Sequential()
    model.add(Dense(64, activation="relu", input_dim=input_dim))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_model(X_train, y_train):
    global model
    model = build_ann(X_train.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)
    
    # Save the trained model
    model.save('asd_model.h5')

def prepare_model_and_data():
    global df, X_test, y_test
    X_scaled, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    train_model(X_train, y_train)
    # Save X_test and y_test for later use
    joblib.dump(X_test, 'X_test.pkl')
    joblib.dump(y_test, 'y_test.pkl')

# Load saved objects
def load_saved_objects():
    global sex_encoder, jaundice_encoder, family_mem_with_asd_encoder, scaler, model, X_test, y_test, feature_columns
    try:
        sex_encoder = joblib.load('sex_encoder.pkl')
        jaundice_encoder = joblib.load('jaundice_encoder.pkl')
        family_mem_with_asd_encoder = joblib.load('family_mem_with_asd_encoder.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        model = load_model('asd_model.h5')  # Ensure load_model is available
        X_test = joblib.load('X_test.pkl')
        y_test = joblib.load('y_test.pkl')
    except FileNotFoundError as e:
        st.write(f"Error loading files: {e}")
        st.write("Please make sure to run the model training and saving code first.")
        st.stop()  # Stop execution if files are missing
    except Exception as e:
        st.write(f"An unexpected error occurred: {e}")
        st.stop()

def predict_asd(input_data):
    # Ensure that all global objects are initialized
    global sex_encoder, jaundice_encoder, family_mem_with_asd_encoder, scaler, feature_columns, model
    
    # Check if all necessary objects are loaded
    if sex_encoder is None or jaundice_encoder is None or family_mem_with_asd_encoder is None or scaler is None or model is None or feature_columns is None:
        st.write("Error: Model or encoders not properly loaded.")
        return None
    
    input_data = pd.DataFrame([input_data])
    
    # Handle missing labels
    def handle_missing_labels(series, encoder):
        if encoder is None or not encoder.classes_.size:  # Check if encoder is None
            return series
        return series.apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
    
    if 'Sex' in input_data.columns:
        input_data["Sex"] = handle_missing_labels(input_data["Sex"], sex_encoder)
    
    if 'Jaundice' in input_data.columns:
        input_data["Jaundice"] = handle_missing_labels(input_data["Jaundice"], jaundice_encoder)
    
    if 'Family_mem_with_ASD' in input_data.columns:
        input_data["Family_mem_with_ASD"] = handle_missing_labels(input_data["Family_mem_with_ASD"], family_mem_with_asd_encoder)
    
    # Transform categorical features
    if sex_encoder is not None:
        input_data["Sex"] = sex_encoder.transform(input_data["Sex"])
    
    if jaundice_encoder is not None:
        input_data["Jaundice"] = jaundice_encoder.transform(input_data["Jaundice"])
    
    if family_mem_with_asd_encoder is not None:
        input_data["Family_mem_with_ASD"] = family_mem_with_asd_encoder.transform(input_data["Family_mem_with_ASD"])
    
    # Transform categorical features with dummy variables
    input_data = pd.get_dummies(input_data, columns=["Ethnicity", "Who completed the test"], drop_first=True)
    
    # Ensure input data has the same columns as the training data
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)
    
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
    "A7": st.selectbox("If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them? (e.g. bringing a favorite toy, hugging)", ["Yes", "No"]),
    "A8": st.selectbox("Does your child show concern if you are upset or sad?", ["Yes", "No"]),
    "A9": st.selectbox("Does your child respond to your smile?", ["Yes", "No"]),
    "A10": st.selectbox("Does your child show emotions when interacting with other children?", ["Yes", "No"]),
}

if st.button("Predict"):
    # Prepare input data
    input_data = {
        "Sex": sex,
        "Age": age_mons,
        "Jaundice": jaundice,
        "Family_mem_with_ASD": family_asd,
        "Ethnicity": ethnicity,
        "Who completed the test": who_completed_test,
        **questions
    }
    
    # Load saved objects (including model and encoders)
    load_saved_objects()
    
    # Predict ASD
    result = predict_asd(input_data)
    
    # Show result
    if result is not None:
        if result > 0.5:
            st.write("The model predicts that ASD traits are likely present.")
        else:
            st.write("The model predicts that ASD traits are unlikely present.")
