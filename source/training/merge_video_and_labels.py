import os
import pandas as pd
import numpy as np

def round_to_nearest_bucket(timestamp, bucket_size=20):
    return (timestamp // bucket_size) * bucket_size

def load_labels(video_number, base_path="/mnt-persist/data"): 
    # Locate the labels CSV file
    raw_path = os.path.join(base_path, str(video_number), "raw")
    labels_files = [f for f in os.listdir(raw_path) if f.endswith("_labels.csv")]
    
    if not labels_files:
        print(f"No labels file found for video {video_number}")
        return None
    
    labels_path = os.path.join(raw_path, labels_files[0])
    labels_df = pd.read_csv(labels_path)
    
    # Ensure the timestamp column exists (creating a synthetic one if missing)
    labels_df["timestamp"] = labels_df.index * 20  # Assume each row represents 20s interval
    labels_df["rounded_timestamp"] = labels_df["timestamp"].apply(round_to_nearest_bucket)
    
    # Keep only the score column
    labels_df = labels_df[["rounded_timestamp", "score"]]
    
    return labels_df

def merge_data(video_features_path):
    # Load video features
    video_df = pd.read_csv(video_features_path)
    video_df["rounded_timestamp"] = video_df["timestamp"].apply(round_to_nearest_bucket)
    
    merged_data = []
    for video_number in video_df["video_number"].unique():
        video_subset = video_df[video_df["video_number"] == video_number]
        labels_df = load_labels(video_number)
        
        if labels_df is None:
            continue
        
        merged_df = video_subset.merge(labels_df, on="rounded_timestamp", how="left")
        
        # Drop unwanted columns
        merged_df = merged_df.drop(columns=[col for col in merged_df.columns if "Unnamed" in col])
        merged_df = merged_df.drop(columns=["rounded_timestamp"], errors='ignore')
        
        merged_data.append(merged_df)
    
    final_df = pd.concat(merged_data, ignore_index=True)
    return final_df

# Path to video features CSV
video_features_csv = "/mnt-persist/data/video_features_one_hot.csv"
merged_df = merge_data(video_features_csv)

# Save the merged dataset
output_path = "/mnt-persist/data/merged_video_labels.csv"
merged_df.to_csv(output_path, index=False)

print(f"Merged dataset saved to {output_path}")
