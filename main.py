def predict_asd(input_data):
    global sex_encoder, jaundice_encoder, family_mem_with_asd_encoder, scaler, feature_columns, model
    
    # Ensure that all global objects are initialized
    if sex_encoder is None or jaundice_encoder is None or family_mem_with_asd_encoder is None or scaler is None or model is None or feature_columns is None:
        st.write("Error: Model or encoders not properly loaded.")
        return None
    
    input_data = pd.DataFrame([input_data])
    
    # Handle missing labels
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
    
    # Debugging: Check the data types and values
    st.write("Input Data Types:")
    st.write(input_data.dtypes)
    
    st.write("Input Data Sample:")
    st.write(input_data.head())
    
    # Convert to float32 before scaling
    try:
        input_data = input_data.astype('float32')
    except ValueError as e:
        st.write(f"Error converting data to float32: {e}")
        return None
    
    # Transform input data using the scaler
    try:
        input_data = scaler.transform(input_data)
    except Exception as e:
        st.write(f"Error scaling data: {e}")
        return None
    
    # Predict
    try:
        prediction = model.predict(input_data)
        return prediction[0][0]
    except Exception as e:
        st.write(f"Error making prediction: {e}")
        return None
