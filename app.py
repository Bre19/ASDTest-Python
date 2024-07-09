import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Dummy dataset and model training for demonstration
def create_dummy_data():
    np.random.seed(42)
    X = pd.DataFrame({
        'A1': np.random.randint(0, 2, size=100),
        'A2': np.random.randint(0, 2, size=100),
        'A3': np.random.randint(0, 2, size=100),
        'A4': np.random.randint(0, 2, size=100),
        'A5': np.random.randint(0, 2, size=100),
        'A6': np.random.randint(0, 2, size=100),
        'A7': np.random.randint(0, 2, size=100),
        'A8': np.random.randint(0, 2, size=100),
        'A9': np.random.randint(0, 2, size=100),
        'A10': np.random.randint(0, 2, size=100),
        'Age_Years': np.random.randint(12, 240, size=100),
        'Sex': np.random.choice(['M', 'F'], size=100),
        'Ethnicity': np.random.choice(['asian', 'white', 'black'], size=100),
        'Jaundice': np.random.choice(['Yes', 'No'], size=100),
        'Family_mem_with_ASD': np.random.choice(['Yes', 'No'], size=100),
        'Who_completed_the_test': np.random.choice(['Parent', 'Health Care Professional', 'Family Member'], size=100)
    })
    y = np.random.randint(0, 2, size=100)
    return X, y

X, y = create_dummy_data()

# Preprocessing
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

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_scaled, y)

# Fungsi untuk menerima input pengguna dari Streamlit
def get_user_input():
    responses = []
    for i in range(1, 11):
        response = st.selectbox(f"Pertanyaan {i} (1 untuk Ya, 0 untuk Tidak): ", [1, 0], key=f'q{i}')
        responses.append(response)

    age_mons = st.number_input("Masukkan umur dalam bulan: ", min_value=1)
    sex = st.selectbox("Masukkan jenis kelamin (M/F): ", ['M', 'F'])
    ethnicity = st.text_input("Masukkan etnis: ")
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

# Fungsi untuk melakukan preprocessing terhadap input pengguna
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

# Fungsi untuk memberikan prediksi
def predict_asd(model, scaler, sex_encoder, jaundice_encoder, family_mem_with_asd_encoder):
    user_input = get_user_input()
    user_input_preprocessed = preprocess_user_input(user_input, scaler, sex_encoder, jaundice_encoder, family_mem_with_asd_encoder)
    prediction = model.predict(user_input_preprocessed)
    result = "Terkena ASD" if prediction[0] == 1 else "Tidak Terkena ASD"
    st.write(f"Hasil prediksi: {result}")

# Menjalankan prediksi
if __name__ == "__main__":
    st.title("Aplikasi Prediksi ASD")
    predict_asd(model, scaler, sex_encoder, jaundice_encoder, family_mem_with_asd_encoder)