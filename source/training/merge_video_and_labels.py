import os
import pandas as pd
import numpy as np

def round_to_nearest_bucket(timestamp, bucket_size=20):
    """Round a timestamp to the nearest bucket.

    Args:
        timestamp (int): The timestamp to be rounded.
        bucket_size (int, optional): The size of the bucket to round to. Defaults to 20.

    Returns:
        int: The timestamp rounded to the nearest bucket.
    """
    return (timestamp // bucket_size) * bucket_size

def load_labels(video_number, base_path="/mnt-persist/data"):
    """Load labels for a given video number from a CSV file.

    This function locates the labels CSV file for a specified video number,
    reads it into a DataFrame, and processes the timestamps.

    Args:
        video_number (int): The video number for which to load labels.
        base_path (str, optional): The base directory path where the data is stored. Defaults to "/mnt-persist/data".

    Returns:
        pd.DataFrame or None: A DataFrame containing the rounded timestamps and scores, or None if no labels file is found.
    """
    raw_path = os.path.join(base_path, str(video_number), "raw")
    labels_files = [f for f in os.listdir(raw_path) if f.endswith("_labels.csv")]
    
    if not labels_files:
        print(f"No labels file found for video {video_number}")
        return None
    
    labels_path = os.path.join(raw_path, labels_files[0])
    labels_df = pd.read_csv(labels_path)
    
    labels_df["timestamp"] = labels_df.index * 20  # Assume each row represents 20s interval
    labels_df["rounded_timestamp"] = labels_df["timestamp"].apply(round_to_nearest_bucket)
    
    labels_df = labels_df[["rounded_timestamp", "score"]]
    
    return labels_df

def merge_data(video_features_path):
    """Merge video features with corresponding labels.

    This function loads video features from a CSV file, rounds their timestamps,
    and merges them with the corresponding labels based on the rounded timestamps.

    Args:
        video_features_path (str): The file path to the video features CSV.

    Returns:
        pd.DataFrame: A DataFrame containing the merged video features and labels.
    """
    video_df = pd.read_csv(video_features_path)
    video_df["rounded_timestamp"] = video_df["timestamp"].apply(round_to_nearest_bucket)
    
    merged_data = []
    for video_number in video_df["video_number"].unique():
        video_subset = video_df[video_df["video_number"] == video_number]
        labels_df = load_labels(video_number)
        
        if labels_df is None:
            continue
        
        merged_df = video_subset.merge(labels_df, on="rounded_timestamp", how="left")
        
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
