# API Documentation - ByborgHackathon Codebase

Welcome to the API documentation for the ByborgHackathon project. This codebase is designed to analyze streamer content and audience engagement by processing both video and audio data, extracting key features, and training an explainable machine learning model.

---

## Overview

This repository is structured into several key components:

- **Data Acquisition**  
  Downloads YouTube videos and their associated live chat data using [yt-dlp](https://github.com/yt-dlp/yt-dlp). This module handles raw data retrieval and initial preprocessing.

- **Feature Extraction**  
  - **Audio Extraction:**  
    - Transcribes audio using OpenAI Whisper.
    - Segments transcripts into sentences with timestamps.
    - Performs sentiment analysis and extracts structured features related to speech (e.g., interruptions, humor, technical content).
  - **Image Extraction:**  
    - Samples video frames at regular intervals.
    - Processes images with a vision-language model to extract visual features (e.g., number of faces, objects, and actions).

- **Label Extraction**  
  Processes chat logs and donation data to compute viewer engagement metrics. This includes calculating message rates, distinct author rates, and donation frequencies, which are then combined into an overall engagement score.

- **Training & Merging**  
  Merges video features with the engagement labels to create a comprehensive dataset. An explainable decision tree model is then trained to identify the most important factors driving engagement.

- **Frontend Visualization**  
  Uses Streamlit to create an interactive dashboard where users can explore engagement metrics, key moments from streams, and feature importance insights. This dashboard integrates data visualizations powered by Plotly.

---

## Repository Structure

The codebase is organized as follows:

```plaintext
├── data_acquisition
├── feature_extraction
│   ├── audio
│   └── image
├── label_extraction
├── training
└── frontend
```



