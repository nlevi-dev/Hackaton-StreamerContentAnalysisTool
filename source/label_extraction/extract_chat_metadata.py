import json
import pandas as pd
import argparse
import os
import datetime
import glob 

def convert_usec_to_seconds(timestamp_usec):
    """ Converts microseconds to seconds and returns an integer. """
    try:
        var = int(timestamp_usec) // 1000000 
        return var  # Convert µs to seconds
    except (ValueError, TypeError):
        return None


def calculate_message_rate(df: pd.DataFrame, min_time: int, max_time: int, window_length: int = 60) -> list[dict]:
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

    Parameters:
    - json_file: Path to the input JSON file.
    - output_json: Path to the output JSON file.
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