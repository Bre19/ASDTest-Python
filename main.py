import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the dataset
df = pd.read_csv('path_to_your_data.csv')

# Fit and save encoders and scaler
def preprocess_data(df):
    df.replace({"Yes": 1, "No": 0}, inplace=True)
    
    global sex_encoder, jaundice_encoder, family_mem_with_asd_encoder, scaler
    
    # Initialize and fit label encoders
    sex_encoder = LabelEncoder()
    jaundice_encoder = LabelEncoder()
    family_mem_with_asd_encoder = LabelEncoder()
    
    sex_encoder.fit(df["Sex"])
    jaundice_encoder.fit(df["Jaundice"])
    family_mem_with_asd_encoder.fit(df["Family_mem_with_ASD"])
    
    X = df.drop("Class/ASD Traits ", axis=1)
    y = df["Class/ASD Traits "]
    
    # Transform categorical features
    X["Sex"] = sex_encoder.transform(X["Sex"])
    X["Jaundice"] = jaundice_encoder.transform(X["Jaundice"])
    X["Family_mem_with_ASD"] = family_mem_with_asd_encoder.transform(X["Family_mem_with_ASD"])
    X = pd.get_dummies(X, columns=["Ethnicity", "Who completed the test"], drop_first=True)
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the encoders and scaler
    joblib.dump(sex_encoder, 'sex_encoder.pkl')
    joblib.dump(jaundice_encoder, 'jaundice_encoder.pkl')
    joblib.dump(family_mem_with_asd_encoder, 'family_mem_with_asd_encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    return X_scaled, y

# Preprocess data and split into training and testing sets
X_scaled, y = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model.pkl')

# Evaluate the model
test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Prediction function
def predict_asd(input_data):
    input_data = pd.DataFrame([input_data])
    
    # Load encoders and scaler
    sex_encoder = joblib.load('sex_encoder.pkl')
    jaundice_encoder = joblib.load('jaundice_encoder.pkl')
    family_mem_with_asd_encoder = joblib.load('family_mem_with_asd_encoder.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Handle missing labels
    def handle_missing_labels(series, encoder):
        return series.apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])

    if 'Sex' in input_data.columns:
        input_data["Sex"] = handle_missing_labels(input_data["Sex"], sex_encoder)
    
    if 'Jaundice' in input_data.columns:
        input_data["Jaundice"] = handle_missing_labels(input_data["Jaundice"], jaundice_encoder)
    
    if 'Family_mem_with_ASD' in input_data.columns:
        input_data["Family_mem_with_ASD"] = handle_missing_labels(input_data["Family_mem_with_ASD"], family_mem_with_asd_encoder)
    
    # Transform categorical features
    input_data["Sex"] = sex_encoder.transform(input_data["Sex"])
    input_data["Jaundice"] = jaundice_encoder.transform(input_data["Jaundice"])
    input_data["Family_mem_with_ASD"] = family_mem_with_asd_encoder.transform(input_data["Family_mem_with_ASD"])
    input_data = pd.get_dummies(input_data, columns=["Ethnicity", "Who completed the test"], drop_first=True)
    
    # Ensure input data has the same columns as the training data
    input_data = input_data.reindex(columns=df.columns.difference(["Class/ASD Traits "]), fill_value=0)
    
    # Scale the data
    input_data = scaler.transform(input_data)
    input_data = input_data.astype('float32')
    
    # Make the prediction
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app
def main():
    st.title("ASD Prediction App")

    # Input fields
    input_data = {
        "Sex": st.selectbox("Sex", ["Male", "Female"]),
        "Jaundice": st.selectbox("Jaundice", ["Yes", "No"]),
        "Family_mem_with_ASD": st.selectbox("Family_member_with_ASD", ["Yes", "No"]),
        "Ethnicity": st.selectbox("Ethnicity", ["Ethnicity1", "Ethnicity2"]),  # Example values
        "Who completed the test": st.selectbox("Who completed the test", ["Parent", "Teacher"]),  # Example values
    }
    
    # Predict
    if st.button("Predict"):
        try:
            prediction = predict_asd(input_data)
            if prediction > 0.5:
                st.write("The model predicts: Likely to have ASD.")
            else:
                st.write("The model predicts: Unlikely to have ASD.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
