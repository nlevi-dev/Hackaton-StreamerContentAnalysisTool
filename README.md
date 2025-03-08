# ByborgHackathon
Codebase for the 24H Byborg AI Hackathon

# Streamer Content Analysis Tool  

## Overview  

This project is a **retrospective analysis tool for streamers**, designed to help creators understand how different aspects of their content impact audience engagement. The system extracts structured features from both **audio and video**, correlates them with **viewer interaction metrics**, and uses an **explainable machine learning model** to uncover patterns that drive engagement.  

## Key Components  

### 1. Feature Extraction  
- **Audio Analysis**:  
  - Transcribes speech and extracts structured metrics using an LLM.  
  - Detects key elements such as:  
    - Frequency of audience engagement.  
    - Mentions of price or cost.  
    - Sponsorship mentions.  

- **Video Analysis**:  
  - Samples frames at regular intervals and processes them with a vision-language model.  
  - Extracts visual features such as:  
    - Number of faces in the frame.  
    - Presence of dramatic poses or actions.  
    - Visible objects (e.g., boxes in unboxing content).  

### 2. Engagement Labeling  
- Extracts audience interaction KPIs, including:  
  - **Chat activity rate**  
  - **Donation frequency**  
  - **Replay heatmap data**  
- These metrics are combined into a **single engagement score**.  

### 3. Explainable Machine Learning Model  
- Uses a **decision tree** to analyze correlations between content features and engagement.  
- Example insight: **Streams with more visible boxes tend to have higher engagement**, aligning with the popularity of unboxing content.  

## Why Use This Tool?  
This tool helps streamers refine their content strategy by providing **clear, data-driven insights** into what keeps their audience engaged.  

# Source
```
├── data_acquisition
├── feature_extraction
│   ├── audio
│   └── image
├── label_extraction
├── training
└── frontend
```
### Run documentation
```bash
poetry install

poetry run mkdocs build

poetry run mkdocs serve
```

### data_acquisition

#### download-data.py
This script downloads YouTube videos and their chat history based on a list of URLs provided in a text file.  
##### Prerequisites  
Make sure you have Python installed and the necessary dependencies (e.g., `argparse`, `yt-dlp`).  
##### Usage  
Run the script from the command line:  
```bash
python download-data.py <input_file> [--cookies_file <path>] [--output_path <path>]
```

#### preprocess-videos.py
This script processes video files by **extracting images** at a specified resolution and frame rate and **extracting audio** in MP3 format. It also ensures correct file permissions and uses multiprocessing to speed up audio processing.  
##### Prerequisites  
Ensure you have **FFmpeg** installed and accessible from the command line.  
##### Usage  
Run the script from the command line:  
```bash
python preprocess-videos.py [base_dir] [--video_resolution <WxH>] [--video_fps <fps>]
```

### feature_extraction - audio

##### Usage  
Run the script from the command line:  
```bash
python extract.py path_to_mp3 [--debug]
```
##### Features
- Transcription: Converts speech to text using OpenAI Whisper.
- Sentence Segmentation: Structures transcript into sentences with timestamps.
- Sentiment Analysis: Classifies each sentence's sentiment.
- Feature Extraction: Identifies specific characteristics such as interruptions, humor, and technical terms.
- Chunk Processing: Breaks transcript into segments for structured data extraction.
##### Output
- Processed transcript with timestamps and sentiment labels (*_audio_raw.npy).
- Extracted feature data saved as pickle files (feature_audio/).
- Debug logs (if enabled) stored in feature_audio_debug/.
##### Requirements
- Python 3.x
- torch, transformers, librosa, numpy, pandas
- Ensure a compatible GPU is available for optimal performance.

### feature_extraction - image

##### Usage  
Run the script from the command line:  
```bash
python extract.py path_to_folder [--debug]
```
##### Features
This script extracts structured features from images using a vision-language model (LLaVA) and saves the results in a structured format.
##### Output
- Processed feature data is saved as .pkl files in feature_video/.
- If --debug is enabled, additional human-readable text files are saved in feature_video_debug/.
##### Requirements
- Python 3.x
- torch, transformers, numpy, pandas, PIL, jaro
- Ensure a compatible GPU is available for optimal performance.

### label_extraction
This script processes chat data from JSON files, extracts messages and donation events, and calculates engagement metrics over time. The output is stored as cleaned JSON and CSV files for further analysis.
##### Usage  
Run the script from the command line:  
```bash
python script.py base_directory [-wl WINDOW_LENGTH]
```
##### Features
- Chat message and donation extraction: Converts raw chat logs into structured JSON format.
- Message rate (messages per time window)
- Distinct authors per window
- Active user rate (ratio of unique authors to total messages)
- Donation rate (number and amount of donations per window)
- Optional integration with metadata heatmaps for scoring high-engagement moments
##### Output
- Cleaned chat data: Saved as <filename>_clean.json inside each raw/ folder.
- Engagement metrics: Saved as <filename>_labels.csv
- Top 5 high-engagement timestamps: Printed in hh:mm:ss format.
##### Requirements
- Python 3.x
- pandas

### training

#### preprocess.py
This script loads serialized data from pickle files, processes video and audio metadata, and applies transformations for further analysis. The output is stored as a structured CSV file.
##### Output
A CSV file containing structured video and audio feature data

#### merge_video_and_labels.py
This script processes video and audio feature data and merges it with precomputed label scores to create a structured dataset for further analysis.
##### Output
A CSV file containing merged video and audio features and label scores

#### train_rules.py
This script processes video engagement data, balances class distribution, and trains a decision tree model to classify engagement levels as "low" or "high".
##### Output
- Decision tree rules printed to the console.
- Top 3 most important features saved to top_features.csv
