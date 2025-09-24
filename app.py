import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="Baby Incubator Monitoring", layout="wide")

st.title("🍼 Neonatal Baby Incubator - Live Monitoring & Prediction")

# Load dataset
df = pd.read_csv("incubator_data.csv")

# Split into features and target
X = df[['temperature', 'humidity', 'heart_rate', 'weight']]
y = df['status']  # 0 = Safe, 1 = Risk

# Train a simple logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Simulation settings
total_time = 10  # minutes
interval = 60    # seconds (1 min)
max_readings = int(total_time * 60 / interval)  # 10 readings in 10 minutes

# Start live simulation
st.subheader("📡 Live Data Stream (Simulated from CSV)")

placeholder = st.empty()

for i in range(max_readings):
    with placeholder.container():
        # Take new data sample
        sample = df.sample(1, random_state=i).reset_index(drop=True)
        temp, hum, hr, wt = sample.iloc[0][['temperature', 'humidity', 'heart_rate', 'weight']]
        
        # Prediction
        pred = model.predict(sample[['temperature', 'humidity', 'heart_rate', 'weight']])[0]
        status = "⚠️ At Risk" if pred == 1 else "✅ Safe"

        st.metric("Temperature (°C)", f"{temp:.1f}")
        st.metric("Humidity (%)", f"{hum:.1f}")
        st.metric("Heart Rate (bpm)", f"{hr:.0f}")
        st.metric("Weight (kg)", f"{wt:.2f}")
        st.write(f"**Prediction:** {status}")

        # Graphs
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))
        axs[0,0].plot(df['temperature'][:i+1]); axs[0,0].set_title("Temperature")
        axs[0,1].plot(df['humidity'][:i+1]); axs[0,1].set_title("Humidity")
        axs[1,0].plot(df['heart_rate'][:i+1]); axs[1,0].set_title("Heart Rate")
        axs[1,1].plot(df['weight'][:i+1]); axs[1,1].set_title("Weight")
        st.pyplot(fig)

        st.info(f"⏳ Waiting {interval} seconds before next reading...")
    
    time.sleep(interval)

st.success("✅ Simulation finished (10 minutes).")










