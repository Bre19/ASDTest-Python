import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Fungsi untuk memuat dataset dari GitHub
@st.cache_data
def load_data():
    url1 = 'https://raw.githubusercontent.com/Bre19/ASDTest-Python/main/data/Toddler%20Autism%20dataset%20July%202018.csv'
    url2 = 'https://raw.githubusercontent.com/Bre19/ASDTest-Python/main/data/autism_screening.csv'
    url3 = 'https://raw.githubusercontent.com/Bre19/ASDTest-Python/main/data/data_csv.csv'
    d1 = pd.read_csv(url1)
    d2 = pd.read_csv(url2)
    d3 = pd.read_csv(url3)
    return d1, d2, d3

# Fungsi untuk preprocessing data
@st.cache_data
def preprocess_data(d1, d2, d3):
    pd.set_option('display.max_columns', None)
    
    # Proses dataset pertama
    d1["Age_Mons"] = (d1["Age_Mons"] / 12).astype(int)
    
    # Proses dataset kedua
    d2 = d2.dropna()
    
    # Proses dataset ketiga
    d3 = d3.dropna()
    d3["age"] = (d3["age"] / 12).astype(int)
    
    # Sinkronisasi kolom dataset
    d1 = d1.iloc[:, 1:]
    d2 = pd.concat([d2.iloc[:, 1:11], d2.iloc[:, [12, 13, 22, 23, 24, 25, 26, 27]]], axis=1)
    d3 = pd.concat([d3.iloc[:, 0:11], d3.iloc[:, [17, 11, 12, 13, 14, 19, 20]]], axis=1)
    
    d1.columns = d2.columns
    d3.columns = d2.columns
    
    data = pd.concat([d1, d2, d3], axis=0)
    
    # Replacing inconsistent values
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
    
    data['Sex'] = data['Sex'].replace(replacements)
    data['Jaundice'] = data['Jaundice'].replace(replacements)
    data['Family_mem_with_ASD'] = data['Family_mem_with_ASD'].replace(replacements)
    data['ASD_traits'] = data['ASD_traits'].replace(replacements)
    data['Ethnicity'] = data['Ethnicity'].replace(replacements)
    data['Who_completed_the_test'] = data['Who_completed_the_test'].replace(replacements)
    
    X = data.drop("ASD_traits", axis=1)
    y = data["ASD_traits"]
    
    return X, y

# Fungsi untuk memproses data pengguna
def process_user_input(user_input, scaler, sex_encoder, jaundice_encoder, family_mem_with_asd_encoder):
    df = pd.DataFrame([user_input])
    
    df = pd.get_dummies(df, columns=["Ethnicity", "Who_completed_the_test"], drop_first=True)
    
    df['Sex'] = sex_encoder.transform(df['Sex'])
    df['Jaundice'] = jaundice_encoder.transform(df['Jaundice'])
    df['Family_mem_with_ASD'] = family_mem_with_asd_encoder.transform(df['Family_mem_with_ASD'])
    
    missing_cols = set(scaler.feature_names_in_).difference(df.columns)
    for col in missing_cols:
        df[col] = 0
    
    df = df[scaler.feature_names_in_]
    
    df_scaled = scaler.transform(df)
    
    return df_scaled

# Load datasets
d1, d2, d3 = load_data()

# Preprocess datasets
X, y = preprocess_data(d1, d2, d3)

# Encode categorical variables
sex_encoder = LabelEncoder().fit(X['Sex'])
jaundice_encoder = LabelEncoder().fit(X['Jaundice'])
family_mem_with_asd_encoder = LabelEncoder().fit(X['Family_mem_with_ASD'])

# Standardize features
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Split data
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train ANN model
model = Sequential()

# Adding input layer
model.add(Dense(64, activation="relu", input_dim=X_scaled.shape[1]))

# Adding First Hidden Layer
model.add(Dense(64, activation="relu"))

# Adding Output Layer
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=10,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=None
)

model.fit(X_train_scaled, y_train, batch_size=20, epochs=100, validation_split=0.2, callbacks=callback)

# Fungsi untuk mendapatkan input pengguna
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
    ethnicity = st.selectbox("Masukkan etnis: ", ['Middle Eastern', 'Mixed', 'Asian', 'Black', 'South Asian', 'Pacifica'])
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

# Fungsi untuk memberikan prediksi menggunakan ANN
def predict_asd(model, scaler, sex_encoder, jaundice_encoder, family_mem_with_asd_encoder):
    user_input = get_user_input()
    user_input_processed = process_user_input(user_input, scaler, sex_encoder, jaundice_encoder, family_mem_with_asd_encoder)
    
    prediction_prob = model.predict(user_input_processed)[0][0]
    prediction = np.where(prediction_prob > 0.5, 'Kemungkinan besar ASD', 'Kemungkinan kecil ASD')
    
    st.write(f"Probabilitas ASD: {prediction_prob:.2f}")
    st.write(f"Prediksi: {prediction}")

# Main app
def main():
    st.title("Aplikasi Prediksi ASD")

    if st.button("Prediksi ASD"):
        predict_asd(model, scaler, sex_encoder, jaundice_encoder, family_mem_with_asd_encoder)

if __name__ == "__main__":
    main()
