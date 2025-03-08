import streamlit as st
import numpy as np
import pandas as pd

chart_data = pd.read_csv("/home/mika/ByborgAI/processed/df2.csv")

st.line_chart(chart_data["score"].rolling(window=5).mean())
        
