import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import datetime
from scipy.signal import argrelextrema
import re 
from sklearn.preprocessing import MinMaxScaler

# Set Streamlit page configuration
st.set_page_config(page_title="Streamer Engagement Dashboard", layout="wide")

def convert_index_to_timestamp(indices):
    """Convert indices to timestamps for better readability."""
    seconds = [idx * 20 for idx in indices]  # Assuming 20-second intervals
    timestamps = [str(datetime.timedelta(seconds=sec)) for sec in seconds]
    return timestamps, seconds  

# Load the datasets
chat_data = pd.read_csv("/mnt-persist/data/1/raw/Our_New_4500_Workstation_PCs_for_Editing.live_chat_labels.csv")
merged_video_labels = pd.read_csv("/mnt-persist/data/merged_video_labels.csv")  # This already has 'video_number'
merged_data = merged_video_labels[merged_video_labels['video_number']==1]

feature_importances = pd.read_csv("./frontend/top_features.csv")

# Compute rolling mean for smoothing
merged_data["rolling_mean"] = merged_data["score"].rolling(window=5).mean()
timestamps, seconds = convert_index_to_timestamp(merged_data.index)
merged_data["timestamps"] = timestamps
merged_data["seconds"] = seconds  # Add raw seconds for YouTube links

# **Detect Local Maxima (Engaging Moments)**
window_size = 5  # Adjustable window for peak detection
merged_data["rolling_mean"] = merged_data["score"].rolling(window=window_size).mean()

# Find local maxima (peaks)
local_max_indices = argrelextrema(merged_data["rolling_mean"].values, np.greater, order=10)[0]

# Filter significant peaks based on a threshold
engagement_threshold = np.percentile(merged_data["rolling_mean"].dropna(), 90)  # Top 10% peaks
key_moments = merged_data.iloc[local_max_indices]
key_moments = key_moments[key_moments["rolling_mean"] >= engagement_threshold]

# **Generate YouTube Links for Engaging Moments**
YOUTUBE_VIDEO_URL = "https://www.youtube.com/watch?v=G4JoDcsk62A&ab_channel=LinusTechTips"
key_moments["youtube_link"] = key_moments["seconds"].apply(lambda sec: f"{YOUTUBE_VIDEO_URL}&t={sec}")

# Title and description
st.title("📊 Streamer Dashboard")
st.markdown("""
**Analyze your engagement metrics easily!**
- See trends in audience interaction
- Identify key moments from your streams
- Click timestamps to jump to the best moments in the video!
""")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    window_size = st.slider("Rolling Mean Window Size", min_value=1, max_value=20, value=5)

    merged_data["rolling_mean"] = merged_data["score"].rolling(window=window_size).mean()
    
    # Update peaks after changing window size
    local_max_indices = argrelextrema(merged_data["rolling_mean"].values, np.greater, order=10)[0]
    key_moments = merged_data.iloc[local_max_indices]
    key_moments = key_moments[key_moments["rolling_mean"] >= engagement_threshold]
    key_moments["youtube_link"] = key_moments["seconds"].apply(lambda sec: f"{YOUTUBE_VIDEO_URL}&t={sec}")

# Create a Plotly figure for engagement score
fig = px.line(merged_data, x="timestamps", y="rolling_mean", labels={"rolling_mean": "Engagement Score"})

# Add engaging moments as clickable hover tooltips
fig.add_scatter(
    x=key_moments["timestamps"], 
    y=key_moments["rolling_mean"], 
    mode="markers", 
    marker=dict(size=10, color="red", symbol="star"),
    name="Most Engaging Moments",
    customdata=key_moments["youtube_link"],  # Custom data for YouTube links
    hovertemplate="<b>Timestamp:</b> %{x}<br>" +
                  "<b>Engagement Score:</b> %{y}<br>" +
                  "<b><a href='%{customdata}' target='_blank'>Watch on YouTube ▶️</a></b><extra></extra>"
)

# Sidebar: Let users choose which features to overlay
with st.sidebar:
    st.header("📊 Feature Selection")
    
    # Extract feature names from the importance file
    top_features = feature_importances.sort_values(by="importance", ascending=False)["feature"].tolist()
    
    # Allow users to select which features to overlay
    selected_features = st.multiselect("Select Features to Overlay:", top_features)

# Overlay selected features
for feature in selected_features:
    if feature in merged_data.columns:
        fig.add_scatter(
            x=merged_data["timestamps"], 
            y=merged_data[feature], 
            mode="lines", 
            name=f"{feature} (Overlay)",
            opacity=0.7  # Slight transparency to differentiate
        )

fig.update_xaxes(tickmode="linear", dtick=int(len(merged_data) / 10))  # Reduce number of ticks

# Display the combined engagement & selected features plot
st.subheader("🔥 Engagement Score & Selected Features Over Time")
st.plotly_chart(fig, use_container_width=True)

# **Display key engaging moments with clickable YouTube links**
st.subheader("🎯 Most Engaging Moments")
for _, row in key_moments.iterrows():
    st.markdown(f"**[{row['timestamps']} - Watch on YouTube ▶️]({row['youtube_link']})**")

# Add a data table
st.subheader("📄 Data Preview")
st.dataframe(merged_data.tail(10))
