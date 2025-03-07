import pandas as pd 


df = pd.read_json("/home/mika/ByborgAI/test_output.json")

# # filter out negative times
# df = df[df["times"][0]!= "-"]

min_time = df["times"].min()
max_time = df["times"].max()

chatmessage_df = df[df["messageType"]=="chatmessage"]

donation_df = df[df["messageType"]=="donation"]

def calculate_message_rate(df: pd.DataFrame, min_time: int, max_time: int, window_length: int = 60, donation_amount: bool = False) -> dict:
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
        active_user_rates[bucket] = distinct_author_rate[bucket]/message_rate[bucket]
    return message_rate, distinct_author_rate, active_user_rates
    

def calculate_donation_rates(df: pd.DataFrame, min_time: int, max_time: int, window_length: int = 60) -> dict:
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




message_rate, distinct_author_rate, active_user_rates = calculate_message_rate(df=chatmessage_df, min_time=min_time, max_time=max_time)
donation_amount_rates, donation_rates = calculate_donation_rates(df=donation_df, min_time=min_time, max_time=max_time)

label_rates = pd.DataFrame()


# Combine into a DataFrame
df = pd.DataFrame.from_dict(
    {
        "message_rate": message_rate,
        "distinct_author_rate": distinct_author_rate,
        "active_user_rates": active_user_rates,
        "donation_amount_rates": donation_amount_rates,
        "donation_rates": donation_rates,
    },
    orient="index"  # Ensures each dictionary is a row
).T  # Transpose so keys become index

df.to_csv("./data/labels.csv")
    
