import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import os
import pandas as pd

csv_filename = "/Users/stevelowe/Documents/your_cat_data.csv"

# If the file doesn't exist, create it
if not os.path.exists(csv_filename):
    print(f"❌ '{csv_filename}' not found! Creating a new one...")

    data = {
        "Time of Day": ["Morning", "Night", "Afternoon", "Evening"],
        "Cat Mood": ["Hyper", "Sleepy", "Playful", "Angry"],
        "Surface": ["Table", "Shelf", "Counter", "Desk"],
        "Human Activity": ["Ignoring Cat", "Watching Netflix", "Cooking", "On Zoom Call"],
        "Object Type": ["Coffee Mug", "Picture Frame", "Spice Jar", "Laptop"]
    }

    df = pd.DataFrame(data)
    df.to_csv(csv_filename, index=False)
    print(f"✅ Created '{csv_filename}' successfully!")

# Load the CSV
df = pd.read_csv(csv_filename)
print(f"📂 Loaded dataset with {len(df)} records.")

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

# User Inputs
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
