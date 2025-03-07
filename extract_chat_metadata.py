import json
import pandas as pd
import argparse
import os

def convert_usec_to_seconds(timestamp_usec):
    """ Converts microseconds to seconds and returns an integer. """
    try:
        var = int(timestamp_usec) // 1000000 
        return var  # Convert µs to seconds
    except (ValueError, TypeError):
        return None

def create_metadata_json(json_file, output_json):
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
                    timestamp_usec = donation_data.get("timestampUsec")
                    timestamp_sec = convert_usec_to_seconds(timestamp_usec) if timestamp_usec else None

                    extracted_data.append({
                        "messageType": "donation",
                        "times": timestamp_sec,
                        "message": donation_data.get("message", {}).get("runs", [{}])[0].get("text", ""),
                        "donationAmount": donation_data.get("purchaseAmountText", {}).get("simpleText", ""),
                        "author": donation_data.get("authorName", {}).get("simpleText", "")
                    })

        except Exception as e:
            print(f"Error processing entry: {e}")

    # Convert to DataFrame and save as json
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=4)

    print(f"Extraction complete. Data saved to {output_json}")

def main():
    parser = argparse.ArgumentParser(description="Extract chat metadata from JSON and save it as a cleaned JSON.")
    parser.add_argument("json_file", type=str, help="Path to the input JSON file")
    parser.add_argument("output_json", type=str, help="Path to the output JSON file")

    args = parser.parse_args()

    create_metadata_json(args.json_file, args.output_json)

if __name__ == "__main__":
    main()
