import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="🍼 Neonatal Incubator Dashboard", layout="wide")

# ------------------ Auto-refresh ------------------
# Refresh every 5 seconds
st_autorefresh(interval=5000, key="incubator_refresh")

# ------------------ Load full CSV ------------------
@st.cache_data
def load_full_data():
    df = pd.read_csv("neonatal_incubator_data_15min.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values('timestamp')
    df['time_idx'] = np.arange(len(df))
    return df

full_df = load_full_data()

# ------------------ Session State for incremental display ------------------
if 'rows_to_show' not in st.session_state:
    st.session_state.rows_to_show = 10  # start with first 10 readings

# Increase rows every refresh
st.session_state.rows_to_show += 10
if st.session_state.rows_to_show > len(full_df):
    st.session_state.rows_to_show = len(full_df)

# Current dataframe to display
df = full_df.iloc[:st.session_state.rows_to_show]

# ------------------ Dashboard ------------------
st.title("🍼 Neonatal Incubator Dashboard (Live Simulation)")
st.markdown("Simulating live data from the incubator. New readings appear every 5 seconds!")

st.subheader("Data Preview")
st.dataframe(df.tail(10))

# ------------------ Parameter Graphs ------------------
st.subheader("📊 Parameter Trends")
st.line_chart(df.set_index('timestamp')[["temperature"]])
st.line_chart(df.set_index('timestamp')[["humidity"]])
st.line_chart(df.set_index('timestamp')[["heart_rate"]])
st.line_chart(df.set_index('timestamp')[["weight"]])

# ------------------ Alerts ------------------
TEMP_LOW, TEMP_HIGH = 36.5, 37.5
HUM_LOW, HUM_HIGH = 50, 65
HR_LOW, HR_HIGH = 120, 160

df['temp_alert'] = ((df['temperature'] < TEMP_LOW) | (df['temperature'] > TEMP_HIGH))
df['hum_alert'] = ((df['humidity'] < HUM_LOW) | (df['humidity'] > HUM_HIGH))
df['hr_alert'] = ((df['heart_rate'] < HR_LOW) | (df['heart_rate'] > HR_HIGH))
df['any_alert'] = df['temp_alert'] | df['hum_alert'] | df['hr_alert']

st.subheader("⚠ Alerts / Status")
for idx, row in df.tail(10).iterrows():
    if row['any_alert']:
        st.markdown(f"<span style='color:red'>⚠ {row['timestamp']} - Temp: {row['temperature']}°C, Hum: {row['humidity']}%, HR: {row['heart_rate']} bpm</span>  **Please check! 🩺**", unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color:green'>✓ {row['timestamp']} - All parameters normal</span>  **Everything is good! 🌟**", unsafe_allow_html=True)

# ------------------ Predictions ------------------
X = df[['time_idx']]

# Weight prediction
y_weight = df['weight']
model_weight = LinearRegression()
model_weight.fit(X, y_weight)
future_idx = np.arange(len(df), len(df)+5).reshape(-1,1)
pred_weight = model_weight.predict(future_idx)
df_pred_weight = pd.DataFrame({"Future Time Index": future_idx.flatten(), "Predicted Weight (kg)": pred_weight})
st.subheader("📈 Predicted Weight Trend")
st.line_chart(df_pred_weight.set_index("Future Time Index"))
if pred_weight[-1] > y_weight.iloc[-1]:
    st.success("📈 Baby is growing well! Keep monitoring. 🍼")
else:
    st.warning("⚠ Baby’s growth is slow. Consult doctor. 🩺")

# Heart rate prediction
y_hr = df['heart_rate']
model_hr = LinearRegression()
model_hr.fit(X, y_hr)
pred_hr = model_hr.predict(future_idx)
df_pred_hr = pd.DataFrame({"Future Time Index": future_idx.flatten(), "Predicted HR (bpm)": pred_hr})
st.subheader("💓 Predicted Heart Rate Trend")
st.line_chart(df_pred_hr.set_index("Future Time Index"))
if pred_hr[-1] < HR_LOW or pred_hr[-1] > HR_HIGH:
    st.warning("💓 Heart rate may go outside safe range. Stay alert!")
else:
    st.info("💓 Heart rate looks stable.")

# Baby progress score
st.subheader("📝 Baby Progress Score")
score = 100
alert_penalty = df['any_alert'].sum() * 2
weight_penalty = 0
if pred_weight[-1] < y_weight.iloc[-1]:
    weight_penalty = 5
final_score = max(score - alert_penalty - weight_penalty, 0)
st.metric("Overall Progress Score", final_score)
if final_score > 90:
    st.balloons()
    st.success("🎉 Excellent! Your baby is doing great!")
elif final_score > 70:
    st.info("🙂 Good. Keep monitoring.")
else:
    st.warning("⚠ Needs attention! Check alerts and consult doctor.")

# Tips
st.subheader("💡 Tips & Info")
st.markdown("""
- Keep temperature between 36.5°C and 37.5°C.  
- Maintain humidity around 50-65%.  
- Regularly check baby’s heart rate and weight.  
- Respond to alerts promptly.  
- This is a guide; **not a replacement for medical advice**.
""")







