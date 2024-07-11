import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st
import os

# Paths for saved files
sex_encoder_path = 'sex_encoder.pkl'
jaundice_encoder_path = 'jaundice_encoder.pkl'
family_mem_with_asd_encoder_path = 'family_mem_with_asd_encoder.pkl'
scaler_path = 'scaler.pkl'
feature_columns_path = 'feature_columns.pkl'
model_path = 'asd_model.h5'
X_test_path = 'X_test.pkl'
y_test_path = 'y_test.pkl'

# Global variables for encoders and model
sex_encoder = None
jaundice_encoder = None
family_mem_with_asd_encoder = None
scaler = None
feature_columns = None
model = None

# Function to load data from URL
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Bre19/ASDTest-Python/main/data/Toddler%20Autism%20dataset%20July%202018.csv"
    df = pd.read_csv(url)
    return df

def preprocess_and_train_model(df):
    df.replace({"Yes": 1, "No": 0}, inplace=True)
    
    # Create and save encoders
    sex_encoder = LabelEncoder()
    jaundice_encoder = LabelEncoder()
    family_mem_with_asd_encoder = LabelEncoder()
    
    sex_encoder.fit(df["Sex"])
    jaundice_encoder.fit(df["Jaundice"])
    family_mem_with_asd_encoder.fit(df["Family_mem_with_ASD"])
    
    joblib.dump(sex_encoder, sex_encoder_path)
    joblib.dump(jaundice_encoder, jaundice_encoder_path)
    joblib.dump(family_mem_with_asd_encoder, family_mem_with_asd_encoder_path)
    
    X = df.drop("Class/ASD Traits ", axis=1)
    y = df["Class/ASD Traits "]
    
    X["Sex"] = sex_encoder.transform(X["Sex"])
    X["Jaundice"] = jaundice_encoder.transform(X["Jaundice"])
    X["Family_mem_with_ASD"] = family_mem_with_asd_encoder.transform(X["Family_mem_with_ASD"])
    
    X = pd.get_dummies(X, columns=["Ethnicity", "Who completed the test"], drop_first=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler and feature columns
    joblib.dump(scaler, scaler_path)
    joblib.dump(X.columns.tolist(), feature_columns_path)
    
    # Train the model
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
        
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)
        
        # Save the trained model
        model.save(model_path)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    train_model(X_train, y_train)
    
    joblib.dump(X_test, X_test_path)
    joblib.dump(y_test, y_test_path)

def load_saved_objects():
    global sex_encoder, jaundice_encoder, family_mem_with_asd_encoder, scaler, feature_columns, model
    
    if os.path.exists(sex_encoder_path):
        sex_encoder = joblib.load(sex_encoder_path)
    else:
        st.write("Encoder file not found. Training model...")
        df = load_data()
        preprocess_and_train_model(df)
        st.write("Model trained and saved.")
        return
    
    if os.path.exists(jaundice_encoder_path):
        jaundice_encoder = joblib.load(jaundice_encoder_path)
    if os.path.exists(family_mem_with_asd_encoder_path):
        family_mem_with_asd_encoder = joblib.load(family_mem_with_asd_encoder_path)
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    if os.path.exists(feature_columns_path):
        feature_columns = joblib.load(feature_columns_path)
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        st.write("Model file not found. Training model...")
        df = load_data()
        preprocess_and_train_model(df)
        st.write("Model trained and saved.")

def predict_asd(input_data):
    global sex_encoder, jaundice_encoder, family_mem_with_asd_encoder, scaler, feature_columns, model
    
    if sex_encoder is None or jaundice_encoder is None or family_mem_with_asd_encoder is None or scaler is None or model is None or feature_columns is None:
        st.write("Error: Model or encoders not properly loaded.")
        return None
    
    input_data = pd.DataFrame([input_data])
    
    input_data["Sex"] = sex_encoder.transform(input_data["Sex"])
    input_data["Jaundice"] = jaundice_encoder.transform(input_data["Jaundice"])
    input_data["Family_mem_with_ASD"] = family_mem_with_asd_encoder.transform(input_data["Family_mem_with_ASD"])
    
    input_data = pd.get_dummies(input_data, columns=["Ethnicity", "Who completed the test"], drop_first=True)
    
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)
    
    try:
        input_data = scaler.transform(input_data)
    except Exception as e:
        st.write(f"Error scaling data: {e}")
        return None
    
    try:
        prediction = model.predict(input_data)
        return prediction[0][0]
    except Exception as e:
        st.write(f"Error making prediction: {e}")
        return None

# Streamlit user interface
st.title("ASD Screening Test")

df = load_data()

# Input user for 10 questions and other features
sex = st.selectbox("Sex", ["Male", "Female"])
age_mons = st.number_input("Age (in months)", min_value=0)
jaundice = st.selectbox("Jaundice", ["Yes", "No"])
family_asd = st.selectbox("Family member with ASD", ["Yes", "No"])
ethnicity = st.selectbox("Ethnicity", df["Ethnicity"].unique() if df is not None else [])
who_completed_test = st.selectbox("Who completed the test", df["Who completed the test"].unique() if df is not None else [])

questions = {
    "A1": st.selectbox("Does your child look at you when you call his/her name?", ["Yes", "No"]),
    "A2": st.selectbox("How easy is it for you to get eye contact with your child?", ["Yes", "No"]),
    "A3": st.selectbox("Does your child point to indicate that s/he wants something? (e.g. a toy that is out of reach)", ["Yes", "No"]),
    "A4": st.selectbox("Does your child show interest in other people or in playing with other children?", ["Yes", "No"]),
    "A5": st.selectbox("Does your child smile in response to your smile?", ["Yes", "No"]),
    "A6": st.selectbox("Does your child bring a favorite toy to show to you?", ["Yes", "No"]),
    "A7": st.selectbox("Does your child respond to your distress (e.g. by bringing a favorite toy, hugging)", ["Yes", "No"]),
    "A8": st.selectbox("Does your child show concern if you are upset or sad?", ["Yes", "No"]),
    "A9": st.selectbox("Does your child respond to your smile?", ["Yes", "No"]),
    "A10": st.selectbox("Does your child show emotions when interacting with other children?", ["Yes", "No"]),
}

if st.button("Predict"):
    input_data = {
        "Sex": sex,
        "Age": age_mons,
        "Jaundice": jaundice,
        "Family_mem_with_ASD": family_asd,
        "Ethnicity": ethnicity,
        "Who completed the test": who_completed_test,
        **{f"A{i+1}": 1 if v == "Yes" else 0 for i, (k, v) in enumerate(questions.items())},
    }
    
    load_saved_objects()
    
    prediction = predict_asd(input_data)
    
    if prediction is not None:
        if prediction > 0.5:
            st.write(f"The model predicts that the individual is likely to have ASD with a probability of {prediction:.2f}")
        else:
            st.write(f"The model predicts that the individual is unlikely to have ASD with a probability of {prediction:.2f}")
