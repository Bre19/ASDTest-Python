import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model
import streamlit as st
import joblib

# Global variables for data and model
df = None
sex_encoder = None
jaundice_encoder = None
family_mem_with_asd_encoder = None
scaler = None
model = None
X_test = None
y_test = None
feature_columns = None

# Function to load data from URL
@st.cache_data
def load_data():
    global df
    url = "https://raw.githubusercontent.com/Bre19/ASDTest-Python/main/data/Toddler%20Autism%20dataset%20July%202018.csv"
    df = pd.read_csv(url)
    return df

# Load saved objects
def load_saved_objects():
    global sex_encoder, jaundice_encoder, family_mem_with_asd_encoder, scaler, model, X_test, y_test, feature_columns
    try:
        sex_encoder = joblib.load('sex_encoder.pkl')
        jaundice_encoder = joblib.load('jaundice_encoder.pkl')
        family_mem_with_asd_encoder = joblib.load('family_mem_with_asd_encoder.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        model = load_model('asd_model.h5')
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
    global sex_encoder, jaundice_encoder, family_mem_with_asd_encoder, scaler, feature_columns, model

    if sex_encoder is None or jaundice_encoder is None or family_mem_with_asd_encoder is None or scaler is None or model is None or feature_columns is None:
        st.write("Error: Model or encoders not properly loaded.")
        return None

    input_data = pd.DataFrame([input_data])

    def handle_missing_labels(series, encoder):
        if encoder is None or not encoder.classes_.size:
            return series
        return series.apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])

    if 'Sex' in input_data.columns:
        input_data["Sex"] = handle_missing_labels(input_data["Sex"], sex_encoder)

    if 'Jaundice' in input_data.columns:
        input_data["Jaundice"] = handle_missing_labels(input_data["Jaundice"], jaundice_encoder)

    if 'Family_mem_with_ASD' in input_data.columns:
        input_data["Family_mem_with_ASD"] = handle_missing_labels(input_data["Family_mem_with_ASD"], family_mem_with_asd_encoder)

    if sex_encoder is not None:
        input_data["Sex"] = sex_encoder.transform(input_data["Sex"])

    if jaundice_encoder is not None:
        input_data["Jaundice"] = jaundice_encoder.transform(input_data["Jaundice"])

    if family_mem_with_asd_encoder is not None:
        input_data["Family_mem_with_ASD"] = family_mem_with_asd_encoder.transform(input_data["Family_mem_with_ASD"])

    input_data = pd.get_dummies(input_data, columns=["Ethnicity", "Who completed the test"], drop_first=True)

    input_data = input_data.reindex(columns=feature_columns, fill_value=0)

    input_data = scaler.transform(input_data)
    input_data = input_data.astype('float32')

    prediction = model.predict(input_data)
    return prediction[0][0]

# Streamlit user interface
st.title("ASD Screening Test")

df = load_data()

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
    "A4": st.selectbox("Does your child point to share interest with you? (e.g. pointing at an interesting sight)", ["Yes", "No"]),
    "A5": st.selectbox("Does your child pretend? (e.g. care for dolls, talk on a toy phone)", ["Yes", "No"]),
    "A6": st.selectbox("Does your child follow where you’re looking?", ["Yes", "No"]),
    "A7": st.selectbox("If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them? (e.g. stroking hair, hugging them)", ["Yes", "No"]),
    "A8": st.selectbox("Does your child notice if you or someone else are upset or angry?", ["Yes", "No"]),
    "A9": st.selectbox("Does your child respond to their name being called?", ["Yes", "No"]),
    "A10": st.selectbox("Does your child follow simple instructions?", ["Yes", "No"])
}

input_data = {
    "Sex": sex,
    "Age": age_mons,
    "Jaundice": jaundice,
    "Family_mem_with_ASD": family_asd,
    "Ethnicity": ethnicity,
    "Who completed the test": who_completed_test,
    **questions
}

if st.button("Predict"):
    load_saved_objects()
    try:
        prediction = predict_asd(input_data)
        if prediction is not None:
            st.write(f"Probability of ASD: {prediction:.2f}")
    except Exception as e:
        st.write(f"Error: {e}")
