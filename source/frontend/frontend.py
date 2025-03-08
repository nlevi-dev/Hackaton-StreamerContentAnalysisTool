import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import datetime
from scipy.signal import argrelextrema
import re 
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set Streamlit page configuration
st.set_page_config(page_title="Streamer Engagement Dashboard", layout="wide")

def convert_index_to_timestamp(indices, wl: int):
    print(indices)
    """Convert indices to timestamps for better readability."""
    seconds = [idx * wl for idx in indices]  # Assuming 20-second intervals
    timestamps = [str(datetime.timedelta(seconds=sec)) for sec in seconds]
    return timestamps, seconds  

# Load the datasets
chat_data = pd.read_csv("/mnt-persist/data/3/raw/The_Ultimate_500_Dollar_Gaming_PC.live_chat_labels.csv")
merged_video_labels = pd.read_csv("/mnt-persist/data/merged_video_labels.csv")  # This already has 'video_number'
merged_data = merged_video_labels[merged_video_labels['video_number']==3]

feature_importances = pd.read_csv("/home/mika/ByborgAI/source/frontend/top_features.csv")

# Compute rolling mean for smoothing
timestamps, seconds = convert_index_to_timestamp(chat_data.index, 20)
chat_data["rolling_mean"] = chat_data["score"].rolling(window=5).mean()
merged_data["rolling_mean"] = merged_data["score"].rolling(window=5).mean()
chat_data["timestamps"] = timestamps
chat_data["seconds"] = seconds
timestamps, seconds = convert_index_to_timestamp(range(len(merged_data)), 60)
merged_data["timestamps"] = timestamps
merged_data["seconds"] = seconds  # Add raw seconds for YouTube links

# **Detect Local Maxima (Engaging Moments)**
window_size = 5  # Adjustable window for peak detection

# Find local maxima (peaks)
local_max_indices = argrelextrema(chat_data["rolling_mean"].values, np.greater, order=10)[0]

# Filter significant peaks based on a threshold
engagement_threshold = np.percentile(chat_data["rolling_mean"].dropna(), 90)  # Top 10% peaks
key_moments = chat_data.iloc[local_max_indices]
key_moments = key_moments[key_moments["rolling_mean"] >= engagement_threshold]

# **Generate YouTube Links for Engaging Moments**
YOUTUBE_VIDEO_URL = "https://www.youtube.com/watch?v=fGXdUp9rfHo"
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

    chat_data["rolling_mean"] = chat_data["score"].rolling(window=window_size).mean()
    
    # Update peaks after changing window size
    local_max_indices = argrelextrema(chat_data["rolling_mean"].values, np.greater, order=10)[0]
    key_moments = chat_data.iloc[local_max_indices]
    key_moments = key_moments[key_moments["rolling_mean"] >= engagement_threshold]
    key_moments["youtube_link"] = key_moments["seconds"].apply(lambda sec: f"{YOUTUBE_VIDEO_URL}&t={sec}")

# Create a Plotly figure for engagement score

# Create a subplot with a secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])
# Add engagement score line (primary y-axis)
fig.add_trace(
    go.Scatter(
        x=chat_data["timestamps"], 
        y=chat_data["rolling_mean"], 
        mode="lines",
        name="Engagement Score",
        line=dict(color="blue")
    ),
    secondary_y=False
)

fig.update_xaxes(tickmode="linear", dtick=int(len(chat_data) / 10))
# Add engaging moments as clickable hover tooltips
# Add engaging moments (scatter points)
fig.add_trace(
    go.Scatter(
        x=key_moments["timestamps"], 
        y=key_moments["rolling_mean"], 
        mode="markers", 
        marker=dict(size=10, color="red", symbol="star"),
        name="Most Engaging Moments",
        customdata=key_moments["youtube_link"],
        hovertemplate="<b>Timestamp:</b> %{x}<br>" +
                      "<b>Engagement Score:</b> %{y}<br>" +
                      "<b><a href='%{customdata}' target='_blank'>Watch on YouTube ▶️</a></b><extra></extra>"
    ),
    secondary_y=False
)

# Sidebar: Let users choose which features to overlay
with st.sidebar:
    st.header("📊 Feature Selection")
    
    # Extract feature names from the importance file
    top_features = feature_importances.sort_values(by="importance", ascending=False)["feature"].tolist()
    
    # Allow users to select which features to overlay
    selected_features = st.multiselect("Select Features to Overlay:", top_features)

# Overlay selected features on secondary y-axis
for feature in selected_features:
    if feature in merged_data.columns:
        fig.add_trace(
            go.Scatter(
                x=merged_data["timestamps"], 
                y=merged_data[feature], 
                mode="lines", 
                name=f"{feature} (Overlay)",
                opacity=0.7
            ),
            secondary_y=True  # Set to secondary y-axis
        )

fig.update_xaxes(tickmode="linear", dtick=int(len(merged_data) / 10))  # Reduce number of ticks

# Display the combined engagement & selected features plot
st.subheader("🔥 Engagement Score & Selected Features Over Time")
# Update layout for dual y-axis
fig.update_layout(
    title="🔥 Engagement Score & Selected Features Over Time",
    xaxis=dict(title="Time"),
    yaxis=dict(title="Engagement Score", color="blue"),
    yaxis2=dict(title="Feature Values", overlaying="y", side="right", color="green"),
    legend=dict(x=0.01, y=1)
)

# Show plot in Streamlit
st.plotly_chart(fig, use_container_width=True)

# **Display key engaging moments with clickable YouTube links**
st.subheader("🎯 Most Engaging Moments")
for _, row in key_moments.iterrows():
    st.markdown(f"**[{row['timestamps']} - Watch on YouTube ▶️]({row['youtube_link']})**")

# Add a data table
st.subheader("📄 Data Preview")
st.dataframe(merged_data.tail(10))
