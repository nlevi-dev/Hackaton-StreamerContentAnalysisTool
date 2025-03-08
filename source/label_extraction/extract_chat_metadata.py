import json
import pandas as pd
import argparse
import os
import datetime
import glob 

def convert_usec_to_seconds(timestamp_usec):
    """Converts a timestamp from microseconds to seconds.

    This function converts a given timestamp from microseconds to seconds by
    performing an integer division by 1,000,000. If the input is not a valid
    integer or cannot be converted to an integer, the function returns None.

    Args:
        timestamp_usec (Union[int, str]): The timestamp in microseconds. It can be
            provided as an integer or a string that represents an integer.

    Returns:
        Optional[int]: The timestamp converted to seconds as an integer. If the
        input is invalid, returns None.

    Raises:
        ValueError: If the input cannot be converted to an integer.
        TypeError: If the input is of an unsupported type.

    Examples:
        >>> convert_usec_to_seconds(1000000)
        1
        >>> convert_usec_to_seconds('2000000')
        2
        >>> convert_usec_to_seconds('invalid')
        None
    """
    try:
        var = int(timestamp_usec) // 1000000 
        return var  # Convert µs to seconds
    except (ValueError, TypeError):
        return None

def calculate_message_rate(df: pd.DataFrame, min_time: int, max_time: int, window_length: int = 60) -> list[dict]:
    """
    Calculate the message rate, distinct author rate, and active user rate for chat messages.

    This function calculates the rate of messages, the rate of distinct authors, and the active user rate
    within specified time windows. The rates are calculated by dividing the total time into buckets of
    a given window length and counting the number of messages and distinct authors in each bucket.

    Args:
        df (pd.DataFrame): The DataFrame containing chat messages with columns "times" and "author".
        min_time (int): The minimum timestamp in seconds.
        max_time (int): The maximum timestamp in seconds.
        window_length (int, optional): The length of each time window in seconds. Defaults to 60 seconds.

    Returns:
        list[dict]: A list containing three dictionaries:
            - message_rate: A dictionary where keys are bucket start times and values are the number of messages.
            - distinct_author_rate: A dictionary where keys are bucket start times and values are the number of distinct authors.
            - active_user_rates: A dictionary where keys are bucket start times and values are the active user rates.

    Example:
        >>> df = pd.DataFrame({
        ...     "times": [0, 30, 60, 90, 120],
        ...     "author": ["user1", "user2", "user1", "user3", "user2"]
        ... })
        >>> min_time = 0
        >>> max_time = 120
        >>> window_length = 60
        >>> calculate_message_rate(df, min_time, max_time, window_length)
        (
            {0: 2, 60: 2},
            {0: 2, 60: 2},
            {0: 1.0, 60: 1.0}
        )
    """
    total_time = max_time - min_time
    bucket_number = total_time // window_length

    buckets =  [min_time + i * window_length for i in range(bucket_number)]
    message_rate = {}
    distinct_author_rate = {}
    active_user_rates = {}
    for bucket in buckets:
        bucket_content = df[(df["times"] >= bucket) & (df["times"] < bucket + window_length)] 

        # Get length
        length = len(bucket_content)
        message_rate[bucket] = length
        
        # Get distinct authors
        distinct_author_rate[bucket] = bucket_content["author"].nunique()
        if message_rate[bucket] == 0:
            active_user_rates[bucket] = 0
        else:
            active_user_rates[bucket] = distinct_author_rate[bucket]/message_rate[bucket]
    return message_rate, distinct_author_rate, active_user_rates
        

def calculate_donation_rates(df: pd.DataFrame, min_time: int, max_time: int, window_length: int = 60) -> list[dict]:
    """
    Calculate the donation amount rate and donation rate for chat messages.

    This function calculates the rate of donation amounts and the rate of donations
    within specified time windows. The rates are calculated by dividing the total time
    into buckets of a given window length and summing the donation amounts and counting
    the number of donations in each bucket.

    Args:
        df (pd.DataFrame): The DataFrame containing donation messages with columns "times" and "donationAmount".
        min_time (int): The minimum timestamp in seconds.
        max_time (int): The maximum timestamp in seconds.
        window_length (int, optional): The length of each time window in seconds. Defaults to 60 seconds.

    Returns:
        list[dict]: A list containing two dictionaries:
            - donation_amount_rates: A dictionary where keys are bucket start times and values are the sum of donation amounts.
            - donation_rates: A dictionary where keys are bucket start times and values are the number of donations.

    Example:
        >>> df = pd.DataFrame({
        ...     "times": [0, 30, 60, 90, 120],
        ...     "donationAmount": [5, 10, 0, 20, 15]
        ... })
        >>> min_time = 0
        >>> max_time = 120
        >>> window_length = 60
        >>> calculate_donation_rates(df, min_time, max_time, window_length)
        (
            {0: 15, 60: 35},
            {0: 2, 60: 2}
        )
    """
    total_time = max_time - min_time
    bucket_number = total_time // window_length

    buckets =  [min_time + i * window_length for i in range(bucket_number)]
    donation_amount_rates = {}
    donation_rates={}
    for bucket in buckets:
        bucket_content = df[(df["times"] >= bucket) & (df["times"] < bucket + window_length)] 
        donation_amount_rates[bucket] = bucket_content["donationAmount"].sum()
        donation_rates[bucket] = len(bucket_content)
    return donation_amount_rates, donation_rates



def create_metadata_json(json_file, output_json, window_length, metadata_path=None, save_folder=None):
    """
    Extract chat messages and donation data from a JSON file and save them as a CSV.

    This function processes a JSON file containing chat messages and donation data,
    extracts relevant information, calculates message and donation rates, and saves
    the results as a CSV file. Optionally, it can also include metadata from an
    additional JSON file to compute a weighted score.

    Args:
        json_file (str): Path to the input JSON file containing chat messages and donation data.
        output_json (str): Path to the output JSON file where extracted data will be saved.
        window_length (int): The length of each time window in seconds for calculating rates.
        metadata_path (str, optional): Path to an additional JSON file containing metadata for computing a weighted score. Defaults to None.
        save_folder (str, optional): Path to the folder where the output CSV file will be saved. Defaults to None.

    Returns:
        None

    Example:
        >>> create_metadata_json("input.json", "output.json", 60, "metadata.json", "processed")
        This will process the input JSON file, calculate rates, and save the results as a CSV file in the "processed" folder.
    """
    if not os.path.exists(json_file):
        print(f"Error: File '{json_file}' not found.")
        return

    with open(json_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    print("Extracting chat data...")

    extracted_data = []

    for entry in data:
        try:
            actions = entry.get("replayChatItemAction", {}).get("actions", [])
            for action in actions:
                chat_item = action.get("addChatItemAction", {}).get("item", {})

                # Check if it's a chat message
                if "liveChatTextMessageRenderer" in chat_item:
                    chat_data = chat_item["liveChatTextMessageRenderer"]
                    if chat_data.get("timestampText", {}).get("simpleText", "")[0] == "-":
                        continue
                    else:
                        timestamp_usec = chat_data.get("timestampUsec")
                        timestamp_sec = convert_usec_to_seconds(timestamp_usec) if timestamp_usec else None

                        extracted_data.append({
                            "messageType": "chatmessage",
                            "times": timestamp_sec,
                            "message": chat_data.get("message", {}).get("runs", [{}])[0].get("text", ""),
                            "donationAmount": None,
                            "author": chat_data.get("authorName", {}).get("simpleText", "")
                        })

                # Check if it's a donation message
                elif "liveChatPaidMessageRenderer" in chat_item:
                    donation_data = chat_item["liveChatPaidMessageRenderer"]
                    if donation_data.get("timestampText", {}).get("simpleText", "")[0] == "-":
                        continue
                    else:
                        timestamp_usec = donation_data.get("timestampUsec")
                        timestamp_sec = convert_usec_to_seconds(timestamp_usec) if timestamp_usec else None

                        amount_str = donation_data.get("purchaseAmountText", {}).get("simpleText", "")

                        extracted_data.append({
                            "messageType": "donation",
                            "times": timestamp_sec,
                            "message": donation_data.get("message", {}).get("runs", [{}])[0].get("text", ""),
                            "donationAmount": amount_str,
                            "author": donation_data.get("authorName", {}).get("simpleText", "")
                        })

        except Exception as e:
            print(f"Error processing entry: {e}")

    # Convert to DataFrame and save as json
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=4)

    print(f"Extraction complete. Data saved to {output_json}")
    print("Extracting rates of messages and donations")
    
    df = pd.read_json(output_json)

    min_time = df["times"].min()
    max_time = df["times"].max()

    chatmessage_df = df[df["messageType"]=="chatmessage"]

    donation_df = df[df["messageType"]=="donation"]

    

    message_rate, distinct_author_rate, active_user_rates = calculate_message_rate(df=chatmessage_df, 
                                                                                min_time=min_time, 
                                                                                max_time=max_time, 
                                                                                window_length=window_length)
    donation_amount_rates, donation_rates = calculate_donation_rates(df=donation_df, 
                                                                    min_time=min_time, 
                                                                    max_time=max_time,
                                                                    window_length=window_length)
    
    # Combine into a DataFrame
    df2 = pd.DataFrame.from_dict(
        {
            "message_rate": message_rate,
            "distinct_author_rate": distinct_author_rate,
            "active_user_rate": active_user_rates,
            # "donation_amount_rate": donation_amount_rates,
            "donation_rate": donation_rates,
        },
        orient="index"  # Ensures each dictionary is a row
    ).T  # Transpose so keys become index

    # Save CSV file in the processed directory
    labels_csv_path = os.path.join(
        save_folder,
        f"{os.path.splitext(os.path.basename(json_file))[0]}_labels.csv"
    )
 
    df2.index = df2.index - min_time


    # Normalize each column using min-max scaling
    df2 = (df2 - df2.min()) / (df2.max() - df2.min())
    
    if metadata_path != None:
        with open(metadata_path) as f:
            metadata = json.load(f)

        heatmap = metadata["heatmap"]

        def get_score_from_heatmap(time, heatmap):
            for dict in heatmap:
                # print(time)
                if (dict['start_time'] <= time) & (time < dict['end_time']):
                    return dict['value']
        
        df2["heatmap_value"] = df2.index.to_series().apply(lambda idx: get_score_from_heatmap(idx, heatmap))
        

        # Compute the weighted score
        df2["score"] = 0.5 * (0.8 * df2["message_rate"] + 0.2 * df2["donation_rate"]) + 0.5 * df2["heatmap_value"]
    else:
        df2["score"] = 0.8 * df2["message_rate"] + 0.2 * df2["donation_rate"]
    
    df2.to_csv(labels_csv_path)
    print(f"Labels saved to {labels_csv_path}")
    # Find the time steps with the highest score
    biggest_rows = df2.nlargest(5, "score")

    # print(biggest_rows["score"])

    for time in biggest_rows.index:
        print(str(datetime.timedelta(seconds=time)))



def main():
    """
    Main function to extract chat metadata from JSON files and save it as cleaned JSON and CSV files.

    This function processes all numbered folders inside the specified base directory, extracts chat messages
    and donation data from JSON files, calculates message and donation rates, and saves the results as cleaned
    JSON and CSV files. Optionally, it can also include metadata from an additional JSON file to compute a
    weighted score.

    Args:
        None

    Returns:
        None

    Example:
        >>> main()
        This will process all numbered folders inside the base directory, extract chat messages and donation data,
        calculate rates, and save the results as cleaned JSON and CSV files.
    """
    parser = argparse.ArgumentParser(description="Extract chat metadata from JSON and save it as a cleaned JSON.")
    parser.add_argument("base_directory", type=str, help="Path to the base directory containing numbered folders")
    parser.add_argument("-wl", type=int, default=20, help="Length of window for calculating rates")
    args = parser.parse_args()

    base_directory = args.base_directory
    window_length = args.wl

    # Iterate over all numbered folders inside the base directory
    for folder in sorted(os.listdir(base_directory)):
        folder_path = os.path.join(base_directory, folder)

        # Check if it's a directory
        if os.path.isdir(folder_path):
            raw_folder = os.path.join(folder_path, "raw")

            # Process each JSON file in the raw folder
            json_files = glob.glob(os.path.join(raw_folder, "*live_chat.json"))
            info_json_files = glob.glob(os.path.join(raw_folder, "*info.json"))
            info_json_file = info_json_files[0] if info_json_files else None  

            for json_file in json_files:
                output_json = os.path.join(
                    raw_folder,
                    f"{os.path.splitext(os.path.basename(json_file))[0]}_clean.json"
                )
                print(f"Processing: {json_file} -> {output_json}")
                create_metadata_json(json_file, output_json, window_length, info_json_file, raw_folder)
    
if __name__ == "__main__":
    main()