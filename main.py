import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

# Load models and preprocessors
cnn_model = tf.keras.models.load_model('cnn_model.h5')
tabular_model = joblib.load('tabular_lung_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Class labels
tabular_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
image_labels = ['Benign', 'Malignant', 'Normal']

st.title("🫁 Lung Cancer Prediction App")
st.markdown("Predict lung cancer severity using patient survey data and chest X-ray image.")

# Tabs for both models
tab1, tab2 = st.tabs(["📋 Tabular Prediction", "🖼️ Image Prediction"])

# --- Tab 1: Tabular Prediction --- #
with tab1:
    st.subheader("Enter Patient Information")

    # Input form
    input_data = {
        'Age': st.slider("Age", 0, 100, 45),
        'Gender': st.selectbox("Gender", ["Male","Female"]),
        'Air Pollution': st.slider("Air Pollution", 1, 10, 5),
        'Alcohol use': st.slider("Alcohol Use", 1, 10, 5),
        'Smoking': st.slider("Smoking", 1, 10, 5),
        'Genetic Risk': st.slider("Genetic Risk", 1, 10, 5),
        'Balanced Diet': st.slider("Balanced Diet", 1, 10, 5),
        'Fatigue': st.slider("Fatigue", 1, 10, 5),
        'Shortness of Breath': st.slider("Shortness of Breath", 1, 10, 5),
        'Wheezing': st.slider("Wheezing", 1, 10, 5),
        'Chest Pain': st.slider("Chest Pain", 1, 10, 5),
        'Dry Cough': st.slider("Dry Cough", 1, 10, 5)
    }

    # Manually encode Gender

    input_df = pd.DataFrame([input_data])
    
    input_df['Gender'] = input_df['Gender'].map({'Male': 1, 'Female': 2})
    # Show input summary
    st.markdown("### Patient Input Summary")
    st.dataframe(input_df)

    if st.button("🔍 Predict Risk Level"):
        # Encode categorical features using saved label encoders
        for col in label_encoders:
            if col in input_df.columns:
                try:
                    input_df[col] = label_encoders[col].transform(input_df[col])
                except ValueError as e:
                    st.error(f"Encoding error in column '{col}': {e}")
                    st.stop()



        # Scale numeric features
        numeric_columns = ['Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Smoking',
                   'Genetic Risk', 'Balanced Diet', 'Fatigue', 'Shortness of Breath',
                   'Wheezing', 'Chest Pain', 'Dry Cough']


        try:
            input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])
        except Exception as e:
            st.error(f"Scaling error: {e}")
            st.stop()

        # Predict
        try:
            pred = tabular_model.predict(input_df)[0]
            prob = tabular_model.predict_proba(input_df)[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        # Label mapping
        tabular_labels = {0: "Low", 1: "Medium", 2: "High"}

        st.subheader("Prediction Result:")
        st.success(f"Risk Level: **{tabular_labels.get(pred, 'Unknown')}**")

        st.markdown("#### Prediction Probabilities")
        st.json({
            'Low': f"{prob[0]*100:.2f}%",
            'Medium': f"{prob[1]*100:.2f}%",
            'High': f"{prob[2]*100:.2f}%"
        })


# --- Tab 2: Image Prediction --- #
with tab2:
    st.subheader("Upload a Chest X-ray Image")
    uploaded_img = st.file_uploader("Choose an image", type=['jpg', 'png', 'jpeg'])

    if uploaded_img is not None:
        try:
            img = Image.open(uploaded_img).convert('RGB')
            img_resized = img.resize((150, 150))
            img_array = np.array(img_resized) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            prediction = cnn_model.predict(img_batch)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)

            st.image(img, caption="Uploaded Chest X-ray", use_column_width=True)
            st.success(f"Prediction: **{image_labels[predicted_class]}** ({confidence*100:.2f}% confidence)")
        except Exception as e:
            st.error(f"Error processing image: {e}")
