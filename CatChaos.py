import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
csv_filename = "your_cat_data.csv"
df = pd.read_csv(csv_filename)

# Encode categorical data
encoder_dict = {}
for col in df.columns[:-1]:  
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoder_dict[col] = encoder

target_encoder = LabelEncoder()
df["Object Type"] = target_encoder.fit_transform(df["Object Type"])

# Train model
X = df.drop(columns=["Object Type"])
y = df["Object Type"]
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Streamlit UI
st.title("🐱 Cat Chaos Predictor 🏠")
st.write("Predict what your cat will knock over next!")

# Expanded dropdown options to match dataset
time_of_day = st.selectbox("Time of Day", encoder_dict["Time of Day"].classes_)
cat_mood = st.selectbox("Cat Mood", encoder_dict["Cat Mood"].classes_)
surface = st.selectbox("Surface", encoder_dict["Surface"].classes_)
human_activity = st.selectbox("Human Activity", encoder_dict["Human Activity"].classes_)

if st.button("Predict Chaos"):
    input_data = pd.DataFrame({
        "Time of Day": [encoder_dict["Time of Day"].transform([time_of_day])[0]],
        "Cat Mood": [encoder_dict["Cat Mood"].transform([cat_mood])[0]],
        "Surface": [encoder_dict["Surface"].transform([surface])[0]],
        "Human Activity": [encoder_dict["Human Activity"].transform([human_activity])[0]],
    })

    prediction = clf.predict(input_data)
    predicted_object = target_encoder.inverse_transform(prediction)[0]
    st.success(f"🔮 Your cat is about to knock over... **{predicted_object}**! 🐱💥")
    prediction = clf.predict(input_data)
    predicted_object = target_encoder.inverse_transform(prediction)[0]
    st.success(f"🔮 Your cat is about to knock over... **{predicted_object}**! 🐱💥")
