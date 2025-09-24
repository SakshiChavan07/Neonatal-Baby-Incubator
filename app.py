import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st

st.set_page_config(page_title="🍼 Neonatal Incubator Dashboard", layout="wide")

# ------------------ Load Data ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("neonatal_incubator_data_15min.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values('timestamp')
    df['time_idx'] = np.arange(len(df))
    return df

df = load_data()

# ------------------ Dashboard Title ------------------
st.title("🍼 Baby Incubator Monitoring Dashboard")
st.write("Overview of neonatal incubator parameters and predictions.")

# ------------------ Data Preview ------------------
st.subheader("Data Preview")
st.dataframe(df.tail(10))  # last 10 readings

# ------------------ Parameter Graphs ------------------
st.subheader("📊 Parameter Trends")

# Temperature
st.line_chart(df.set_index('timestamp')[["temperature"]])
# Humidity
st.line_chart(df.set_index('timestamp')[["humidity"]])
# Heart Rate
st.line_chart(df.set_index('timestamp')[["heart_rate"]])
# Weight
st.line_chart(df.set_index('timestamp')[["weight"]])

# ------------------ Threshold Alerts ------------------
TEMP_LOW, TEMP_HIGH = 36.5, 37.5
HUM_LOW, HUM_HIGH = 50, 65
HR_LOW, HR_HIGH = 120, 160

df['temp_alert'] = ((df['temperature'] < TEMP_LOW) | (df['temperature'] > TEMP_HIGH))
df['hum_alert'] = ((df['humidity'] < HUM_LOW) | (df['humidity'] > HUM_HIGH))
df['hr_alert'] = ((df['heart_rate'] < HR_LOW) | (df['heart_rate'] > HR_HIGH))
df['any_alert'] = df['temp_alert'] | df['hum_alert'] | df['hr_alert']

st.subheader("⚠ Alerts")
for idx, row in df.tail(20).iterrows():  # last 20 readings
    if row['any_alert']:
        st.markdown(f"<span style='color:red'>⚠ {row['timestamp']} - Temp: {row['temperature']}°C, Hum: {row['humidity']}%, HR: {row['heart_rate']} bpm</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color:green'>✓ {row['timestamp']} - All parameters normal</span>", unsafe_allow_html=True)

# ------------------ Weight Prediction ------------------
st.subheader("📈 Predicted Weight Trend")
X = df[['time_idx']]
y_weight = df['weight']
model_weight = LinearRegression()
model_weight.fit(X, y_weight)

future_idx = np.arange(len(df), len(df)+10).reshape(-1,1)
pred_weight = model_weight.predict(future_idx)

df_pred_weight = pd.DataFrame({
    "Future Time Index": future_idx.flatten(),
    "Predicted Weight (kg)": pred_weight
})
st.line_chart(df_pred_weight.set_index("Future Time Index"))

# ------------------ Heart Rate Prediction ------------------
st.subheader("💓 Predicted Heart Rate Trend")
y_hr = df['heart_rate']
model_hr = LinearRegression()
model_hr.fit(X, y_hr)
pred_hr = model_hr.predict(future_idx)

df_pred_hr = pd.DataFrame({
    "Future Time Index": future_idx.flatten(),
    "Predicted HR (bpm)": pred_hr
})
st.line_chart(df_pred_hr.set_index("Future Time Index"))

# ------------------ Baby Progress Score ------------------
st.subheader("📝 Baby Progress Score")
score = 100
alert_penalty = df['any_alert'].sum() * 2
weight_penalty = 0
if pred_weight[-1] < y_weight.iloc[-1]:
    weight_penalty = 5
final_score = max(score - alert_penalty - weight_penalty, 0)
st.metric("Overall Progress Score", final_score)


"""
import serial
import time
import streamlit as st
import pandas as pd

st.title("🍼 Live Baby Incubator Monitoring (Arduino)")

ser = serial.Serial("COM3", 9600, timeout=1)  # Replace COM3 with your port

data_log = []

for i in range(10):  # 10 readings (10 minutes if delay=1min in Arduino)
    line = ser.readline().decode().strip()
    if line:
        try:
            temp, hum, hr, wt = map(float, line.split(","))
            data_log.append([temp, hum, hr, wt])
            
            df = pd.DataFrame(data_log, columns=["temperature","humidity","heart_rate","weight"])
            st.write(df.tail())

            # Show live graphs
            st.line_chart(df)

        except:
            pass
    
    time.sleep(60)  # wait 1 minute

"""
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


"""







