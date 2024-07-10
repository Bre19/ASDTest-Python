import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Dataset
@st.cache_data
def load_data():
    df = pd.read_csv('ASDTest-Python/data/Toddler Autism dataset July 2018.csv')
    return df

#Preprocess dataset
@st.cache_data
def preprocess_data(df):
    # Assuming the columns are the same as those in the dummy data example
    sex_encoder = LabelEncoder()
    df['Sex'] = sex_encoder.fit_transform(df['Sex'])

    jaundice_encoder = LabelEncoder()
    df['Jaundice'] = jaundice_encoder.fit_transform(df['Jaundice'])

    family_mem_with_asd_encoder = LabelEncoder()
    df['Family_mem_with_ASD'] = family_mem_with_asd_encoder.fit_transform(df['Family_mem_with_ASD'])

    df = pd.get_dummies(df, columns=["Ethnicity", "Who_completed_the_test"], drop_first=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop(columns=['Class/ASD']))  # Assuming 'Class/ASD' is the target column
    y = df['Class/ASD'].values
    
    return X_scaled, y, scaler, sex_encoder, jaundice_encoder, family_mem_with_asd_encoder

df = load_data()
X_scaled, y, scaler, sex_encoder, jaundice_encoder, family_mem_with_asd_encoder = preprocess_data(df)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_scaled, y)

#Inputan user
def get_user_input():
    responses = []
    questions = [
        "Does your child look at you when you call his/her name?",
        "How easy is it for you to get eye contact with your child?",
        "Does your child point to indicate that s/he wants something? (e.g. a toy that is out of reach)",
        "Does your child point to share interest with you? (e.g. pointing at an interesting sight)",
        "Does your child pretend? (e.g. care for dolls, talk on a toy phone)",
        "Does your child follow where you’re looking?",
        "If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them? (e.g. stroking hair, hugging them)",
        "Would you describe your child’s first words as:",
        "Does your child use simple gestures? (e.g. wave goodbye)",
        "Does your child stare at nothing with no apparent purpose?"
    ]
    
    for i, question in enumerate(questions, 1):
        response = st.selectbox(f"{question} (1 untuk Ya, 0 untuk Tidak): ", [1, 0], key=f'q{i}')
        responses.append(response)

    age_mons = st.number_input("Masukkan umur dalam bulan: ", min_value=1)
    sex = st.selectbox("Masukkan jenis kelamin (M/F): ", ['M', 'F'])
    ethnicity = st.selectbox("Masukkan etnis: ", ['asian', 'white', 'black'])
    jaundice = st.selectbox("Apakah Anda pernah mengalami penyakit kuning? (Yes/No): ", ['Yes', 'No'])
    family_mem_with_asd = st.selectbox("Apakah ada anggota keluarga yang memiliki ASD? (Yes/No): ", ['Yes', 'No'])
    who_completed_the_test = st.selectbox("Siapa yang mengisi tes? (Parent/Health Care Professional/Family Member): ", ['Parent', 'Health Care Professional', 'Family Member'])

    user_input = {
        'A1': responses[0],
        'A2': responses[1],
        'A3': responses[2],
        'A4': responses[3],
        'A5': responses[4],
        'A6': responses[5],
        'A7': responses[6],
        'A8': responses[7],
        'A9': responses[8],
        'A10': responses[9],
        'Age_Mons': age_mons,
        'Sex': sex,
        'Ethnicity': ethnicity,
        'Jaundice': jaundice,
        'Family_mem_with_ASD': family_mem_with_asd,
        'Who_completed_the_test': who_completed_the_test
    }

    return user_input

#Preprocess
def preprocess_user_input(user_input, scaler, sex_encoder, jaundice_encoder, family_mem_with_asd_encoder):
    df = pd.DataFrame([user_input])

    df = pd.get_dummies(df, columns=["Ethnicity", "Who_completed_the_test"], drop_first=True)

    df['Sex'] = sex_encoder.transform([df['Sex'][0]])[0]
    df['Jaundice'] = jaundice_encoder.transform([df['Jaundice'][0]])[0]
    df['Family_mem_with_ASD'] = family_mem_with_asd_encoder.transform([df['Family_mem_with_ASD'][0]])[0]

    df.rename(columns={'Age_Mons': 'Age_Years'}, inplace=True)

    missing_cols = set(scaler.feature_names_in_).difference(df.columns)
    for col in missing_cols:
        df[col] = 0

    df = df[scaler.feature_names_in_]

    df_scaled = scaler.transform(df)

    return df_scaled

#Fungsi untuk memberikan prediksi
def predict_asd(model, scaler, sex_encoder, jaundice_encoder, family_mem_with_asd_encoder):
    user_input = get_user_input()
    user_input_preprocessed = preprocess_user_input(user_input, scaler, sex_encoder, jaundice_encoder, family_mem_with_asd_encoder)
    prediction = model.predict(user_input_preprocessed)
    result = "Terkena ASD" if prediction[0] == 1 else "Tidak Terkena ASD"
    st.write(f"Hasil prediksi: {result}")

#Menjalankan prediksi
if __name__ == "__main__":
    st.title("Aplikasi Prediksi ASD")
    predict_asd(model, scaler, sex_encoder, jaundice_encoder, family_mem_with_asd_encoder)
