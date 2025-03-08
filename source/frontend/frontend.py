import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import datetime
from scipy.signal import argrelextrema

# Set Streamlit page configuration
st.set_page_config(page_title="Streamer Engagement Dashboard", layout="wide")

def convert_index_to_timestamp(indices):
    """Convert indices to timestamps for better readability."""
    seconds = [idx * 20 for idx in indices]  # Assuming 20-second intervals
    timestamps = [str(datetime.timedelta(seconds=sec)) for sec in seconds]
    return timestamps, seconds  # Return both formatted timestamps and raw seconds

# Load the data
chart_data = pd.read_csv("/mnt-persist/data/1/raw/Our_New_4500_Workstation_PCs_for_Editing.live_chat_labels.csv")

# Compute rolling mean for smoothing
chart_data["rolling_mean"] = chart_data["score"].rolling(window=5).mean()
timestamps, seconds = convert_index_to_timestamp(chart_data.index)
chart_data["timestamps"] = timestamps
chart_data["seconds"] = seconds  # Add raw seconds for YouTube links

# **Detect Local Maxima (Engaging Moments)**
window_size = 5  # Adjustable window for peak detection
chart_data["rolling_mean"] = chart_data["score"].rolling(window=window_size).mean()

# Find local maxima (peaks)
local_max_indices = argrelextrema(chart_data["rolling_mean"].values, np.greater, order=10)[0]

# Filter significant peaks based on a threshold
engagement_threshold = np.percentile(chart_data["rolling_mean"].dropna(), 90)  # Top 10% peaks
key_moments = chart_data.iloc[local_max_indices]
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

# Create a Plotly figure
fig = px.line(chart_data, x="timestamps", y="rolling_mean", labels={"rolling_mean": "Engagement Score"})

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

fig.update_xaxes(tickmode="linear", dtick=int(len(chart_data) / 10))  # Reduce number of ticks

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    window_size = st.slider("Rolling Mean Window Size", min_value=1, max_value=20, value=5)
    chart_data["rolling_mean"] = chart_data["score"].rolling(window=window_size).mean()
    
    # Update peaks after changing window size
    local_max_indices = argrelextrema(chart_data["rolling_mean"].values, np.greater, order=10)[0]
    key_moments = chart_data.iloc[local_max_indices]
    key_moments = key_moments[key_moments["rolling_mean"] >= engagement_threshold]
    key_moments["youtube_link"] = key_moments["seconds"].apply(lambda sec: f"{YOUTUBE_VIDEO_URL}&t={sec}")

    # Update figure
    fig = px.line(chart_data, x="timestamps", y="rolling_mean", labels={"rolling_mean": "Engagement Score"})
    fig.add_scatter(
        x=key_moments["timestamps"], 
        y=key_moments["rolling_mean"], 
        mode="markers", 
        name="Highest engagement",
        marker=dict(size=10, color="red", symbol="star"),
        customdata=key_moments["youtube_link"],
        hovertemplate="<b>Timestamp:</b> %{x}<br>" +
                      "<b>Engagement Score:</b> %{y}<br>"
    )
    fig.update_xaxes(tickmode="linear", dtick=int(len(chart_data) / 10))

# Display the plot
st.subheader("🔥 Audience Hype Score")
st.plotly_chart(fig, use_container_width=True)

# **Display key engaging moments with clickable YouTube links**
st.subheader("🎯 Most Engaging Moments")
for _, row in key_moments.iterrows():
    st.markdown(f"**[{row['timestamps']} - Watch on YouTube ▶️]({row['youtube_link']})**")

# Add a data table
st.subheader("📄 Data Preview")
st.dataframe(chart_data.tail(10))
