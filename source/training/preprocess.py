import pickle
import os
import pandas as pd


def pickle_load(path):
    """Load a pickle file from the specified path.

    Args:
        path (str): The file path to the pickle file.

    Returns:
        Any: The object loaded from the pickle file.
    """
    with open(path, 'rb') as f:
        ret = pickle.load(f)
    return ret


def load_pickles_from_directory(directory_path: str, video_number: int) -> pd.DataFrame:
    """Load pickle files from a directory and compile them into a DataFrame.

    This function reads all pickle files in the specified directory, extracts
    data from each file, and compiles the data into a pandas DataFrame with
    predefined column names.

    Args:
        directory_path (str): The path to the directory containing pickle files.
        video_number (int): The video number to associate with each entry.

    Returns:
        pd.DataFrame: A DataFrame containing the compiled data from the pickle files.
    """
    column_names = [
        "video_number", "timestamp", "wearing", "actions", "num_people", "num_standing", "num_sitting", "num_looking_at_camera",
        "num_smiling", "num_using_tools", "num_talking", "num_handling_components", "num_reacting_emotionally", "num_gesturing",
        "components", "tools", "num_components", "num_tools", "num_monitors", "num_keyboards", "num_mice", "num_cables", 
        "num_rgb_lights", "num_tables", "num_cases", "num_boxes", "pc_completed", "pc_disassembled", "has_brand_logos", "workspace_clutter",
        "num_faces_visible", "num_text_elements", "num_dominant_colors", "is_closeup", "has_text_overlay", "camera_angle",
        "num_laughing", "num_explaining", "has_exaggerated_expressions", "has_pointing_gesture", "has_surprised_expression", "has_dramatic_pose",
        "visible_brands", "num_brand_logos", "has_ltt_logo", "has_sponsor_logo",
        "num_text_elements", "num_overlay_graphics", "has_subtitle", "has_highlighted_elements", "has_thumbnail_reaction"
    ]
    
    results = {}
    for i, file in enumerate(os.listdir(directory_path)):
        timestamp = int(file[:6])
        results[i] = [video_number] + [timestamp] + pickle_load(os.path.join(directory_path, file))
    
    df = pd.DataFrame.from_dict(results, orient='index', columns=column_names)
    return df


def load_all_videos() -> pd.DataFrame:
    """Load data from all video directories and compile into a single DataFrame.

    This function iterates over a predefined list of video numbers, loads data
    from each video's directory, and concatenates the data into a single DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing data from all specified video directories.
    """
    finished_videos = [i + 1 for i in range(24) if i + 1 not in [14, 15, 17]]
    df_list = []
    for i in finished_videos:
        df_list.append(load_pickles_from_directory(f"/mnt-persist/data/{i}/feature_video", i))
    return pd.concat(df_list, ignore_index=True)


def one_hot_encode_list_columns(df: pd.DataFrame, list_columns: list) -> pd.DataFrame:
    """One-hot encode specified columns containing lists of strings.

    This function drops the specified columns from the DataFrame as the one-hot
    encoding results in too many columns.

    Args:
        df (pd.DataFrame): The DataFrame containing the columns to be encoded.
        list_columns (list): A list of column names to be one-hot encoded.

    Returns:
        pd.DataFrame: The DataFrame with specified columns dropped.
    """
    for col in list_columns:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


Berlinetta = load_all_videos()
# Example usage
list_columns = ["wearing", "actions", "components", "tools", "visible_brands"]  # Adjust based on your actual data
Berlinetta = one_hot_encode_list_columns(Berlinetta, list_columns)
print(Berlinetta)
print(Berlinetta.iloc[0])
Berlinetta.to_csv("/mnt-persist/data/video_features_one_hot.csv")