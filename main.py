import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier

# Load datasets
def load_data():
    d1 = pd.read_csv("https://raw.githubusercontent.com/Bre19/ASDTest-Python/main/data/Toddler%20Autism%20dataset%20July%202018.csv")
    d2 = pd.read_csv("https://raw.githubusercontent.com/Bre19/ASDTest-Python/main/data/autism_screening.csv")
    d3 = pd.read_csv("https://raw.githubusercontent.com/Bre19/ASDTest-Python/main/data/data_csv.csv")
    return d1, d2, d3

# Preprocess data
@st.cache_data
def preprocess_data(d1, d2, d3):
    # Rename columns for consistency
    d1 = d1.rename(columns={
        'Case_No': 'Case_No',
        'Age_Mons': 'Age',
        'Sex': 'Sex',
        'Ethnicity': 'Ethnicity',
        'Jaundice': 'Jaundice',
        'Family_mem_with_ASD': 'Family_mem_with_ASD',
        'Who completed the test': 'Who_completed_the_test',
        'Class/ASD Traits ': 'ASD_traits'
    })
    
    d2 = d2.rename(columns={
        'A1_Score': 'A1',
        'A2_Score': 'A2',
        'A3_Score': 'A3',
        'A4_Score': 'A4',
        'A5_Score': 'A5',
        'A6_Score': 'A6',
        'A7_Score': 'A7',
        'A8_Score': 'A8',
        'A9_Score': 'A9',
        'A10_Score': 'A10',
        'age': 'Age',
        'gender': 'Sex',
        'ethnicity': 'Ethnicity',
        'jundice': 'Jaundice',
        'austim': 'ASD_traits',
        'contry_of_res': 'Ethnicity',
        'used_app_before': 'Who_completed_the_test',
        'result': 'ASD_traits',
        'age_desc': 'Age',
        'relation': 'Family_mem_with_ASD'
    })
    
    d3 = d3.rename(columns={
        'CASE_NO_PATIENT\'S': 'Case_No',
        'A1': 'A1',
        'A2': 'A2',
        'A3': 'A3',
        'A4': 'A4',
        'A5': 'A5',
        'A6': 'A6',
        'A7': 'A7',
        'A8': 'A8',
        'A9': 'A9',
        'A10_Autism_Spectrum_Quotient': 'A10',
        'Social_Responsiveness_Scale': 'Social_Responsiveness_Scale',
        'Age_Years': 'Age',
        'Qchat_10_Score': 'Qchat_10_Score',
        'Speech Delay/Language Disorder': 'Speech_Delay/Language_Disorder',
        'Learning disorder': 'Learning_disorder',
        'Genetic_Disorders': 'Genetic_Disorders',
        'Depression': 'Depression',
        'Global developmental delay/intellectual disability': 'Global_developmental_delay/intellectual_disability',
        'Social/Behavioural Issues': 'Social/Behavioural_Issues',
        'Childhood Autism Rating Scale': 'Childhood_Autism_Rating_Scale',
        'Anxiety_disorder': 'Anxiety_disorder',
        'Sex': 'Sex',
        'Ethnicity': 'Ethnicity',
        'Jaundice': 'Jaundice',
        'Family_mem_with_ASD': 'Family_mem_with_ASD',
        'Who_completed_the_test': 'Who_completed_the_test',
        'ASD_traits': 'ASD_traits'
    })

    # Combine datasets on common columns
    combined_data = pd.concat([d1, d2, d3], axis=0, ignore_index=True, join='inner')

    # Handle missing values
    combined_data = combined_data.dropna()

    # Process categorical variables
    categorical_columns = ['Sex', 'Jaundice', 'Family_mem_with_ASD', 'ASD_traits']
    for col in categorical_columns:
        combined_data[col] = combined_data[col].map({
            'f': 'F', 'm': 'M',
            'yes': 'Yes', 'no': 'No',
            'YES': 'Yes', 'NO': 'No',
            'middle eastern': 'Middle Eastern',
            'Middle Eastern ': 'Middle Eastern',
            'mixed': 'Mixed',
            'asian': 'Asian',
            'black': 'Black',
            'south asian': 'South Asian',
            'PaciFica': 'Pacifica',
            'Pasifika': 'Pasifika',
            'Hispanic': 'Hispanic'
        }).fillna(combined_data[col])

    # Encode categorical columns
    le_sex = LabelEncoder()
    le_jaundice = LabelEncoder()
    le_family_mem_with_asd = LabelEncoder()
    le_asd_traits = LabelEncoder()

    combined_data['Sex'] = le_sex.fit_transform(combined_data['Sex'])
    combined_data['Jaundice'] = le_jaundice.fit_transform(combined_data['Jaundice'])
    combined_data['Family_mem_with_ASD'] = le_family_mem_with_asd.fit_transform(combined_data['Family_mem_with_ASD'])
    combined_data['ASD_traits'] = le_asd_traits.fit_transform(combined_data['ASD_traits'])

    # Features and target variable
    X = combined_data.drop('ASD_traits', axis=1)
    y = combined_data['ASD_traits']

    # Scaling features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, le_sex, le_jaundice, le_family_mem_with_asd, le_asd_traits

# Streamlit app
def main():
    st.title("ASD Prediction Dashboard")

    # Load and preprocess data
    d1, d2, d3 = load_data()
    X_scaled, y, scaler, le_sex, le_jaundice, le_family_mem_with_asd, le_asd_traits = preprocess_data(d1, d2, d3)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train model
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Display metrics
    st.write("Model Evaluation Metrics:")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
    st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
    st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")

    # User input form
    st.sidebar.header("Input Data")

    # Collecting user inputs
    user_responses = []
    questions = [
        "Does your child look at you when you call his/her name?",
        "How easy is it for you to get eye contact with your child?",
        "Does your child use simple gestures? (e.g. wave goodbye)",
        "Does your child stare at nothing with no apparent purpose?"
    ]

    for i, question in enumerate(questions):
        response = st.sidebar.selectbox(f"Question {i+1}: {question}", [1, 0], key=f'q{i+1}')
        user_responses.append(response)

    age_years = st.sidebar.number_input("Enter age in years: ", min_value=1)
    sex = st.sidebar.selectbox("Select gender (M/F): ", ['M', 'F'])
    ethnicity = st.sidebar.selectbox("Select ethnicity: ", ['asian', 'white', 'black'])
    jaundice = st.sidebar.selectbox("Has jaundice ever been experienced? (Yes/No): ", ['Yes', 'No'])
    family_mem_with_asd = st.sidebar.selectbox("Is there a family member with ASD? (Yes/No): ", ['Yes', 'No'])
    who_completed_the_test = st.sidebar.selectbox("Who completed the test? (Parent/Health Care Professional/Family Member): ", ['Parent', 'Health Care Professional', 'Family Member'])

    user_input = user_responses + [age_years, sex, ethnicity, jaundice, family_mem_with_asd, who_completed_the_test]

    # Preprocess user input
    def preprocess_user_input(user_input):
        columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Age', 'Sex', 'Ethnicity_asian', 'Ethnicity_white', 'Who_completed_the_test_Health Care Professional', 'Who_completed_the_test_Family Member']
        df = pd.DataFrame([user_input], columns=columns)

        # Encode categorical features
        df['Sex'] = le_sex.transform(df['Sex'])
        df['Jaundice'] = le_jaundice.transform(df['Jaundice'])
        df['Family_mem_with_ASD'] = le_family_mem_with_asd.transform(df['Family_mem_with_ASD'])

        df = pd.get_dummies(df, columns=['Ethnicity', 'Who_completed_the_test'], drop_first=True)
        
        # Align columns with the training data
        df = df.reindex(columns=X.columns, fill_value=0)

        # Scaling
        df_scaled = scaler.transform(df)

        return df_scaled

    # Predict ASD
    def predict_asd():
        user_input_preprocessed = preprocess_user_input(user_input)
        prediction = model.predict(user_input_preprocessed)
        result = "ASD Detected" if prediction[0] == 1 else "No ASD Detected"
        st.write(f"Prediction Result: {result}")

    # Display prediction result
    if st.sidebar.button("Predict"):
        predict_asd()

if __name__ == "__main__":
    main()
