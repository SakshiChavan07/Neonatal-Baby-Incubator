import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import streamlit as st

# Load dataset from local file
df = pd.read_csv("neonatal_incubator_data_15min.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp")
df["time_idx"] = np.arange(len(df))

# Show data
st.title("🍼 Baby Incubator Monitoring Dashboard")
st.write("Overview of neonatal incubator parameters and predictions.")

st.subheader("Data Preview")
st.write(df.head())

# Plot trends
st.subheader("Parameter Trends")
st.line_chart(df.set_index("timestamp")[["temperature","humidity","heart_rate","weight"]])

# Weight prediction
X = df[['time_idx']]
y = df['weight']
model = LinearRegression()
model.fit(X, y)
future_idx = np.arange(len(df), len(df)+10).reshape(-1,1)
pred_weight = model.predict(future_idx)

st.subheader("📈 Predicted Baby Weight")
st.line_chart(pd.DataFrame({"future_idx": future_idx.flatten(), "Predicted Weight": pred_weight}).set_index("future_idx"))

# Heart rate prediction
y_hr = df['heart_rate']
model_hr = LinearRegression()
model_hr.fit(X, y_hr)
future_hr = model_hr.predict(future_idx)

st.subheader("💓 Predicted Heart Rate")
st.line_chart(pd.DataFrame({"future_idx": future_idx.flatten(), "Predicted HR": future_hr}).set_index("future_idx"))
