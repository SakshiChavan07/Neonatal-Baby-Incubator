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
