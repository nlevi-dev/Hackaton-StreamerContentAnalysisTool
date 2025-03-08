import yt_dlp
import os
import argparse
import stat
import re
import multiprocessing
import time

def sanitize_filename(filename):
    """
    Sanitize the filename by replacing spaces with underscores and removing special characters.

    Parameters:
    filename (str): The original filename to be sanitized.

    Returns:
    str: The sanitized filename.
    """
    # Replace spaces with underscores and remove special characters
    filename = filename.replace(' ', '_')
    return re.sub(r'[^a-zA-Z0-9_]', '', filename)

def download_video_and_chat(url, output_path, index, cookies_file):
    """
    Download a YouTube video and its live chat, saving them to a specified directory.

    Parameters:
    url (str): The URL of the YouTube video.
    output_path (str): The base directory where the video and chat will be saved.
    index (int): An index used to create a unique directory for each video.
    cookies_file (str): Path to the cookies file for authentication.

    Returns:
    None
    """
    # Create a directory for each video using an integer index
    video_dir = os.path.join(output_path, str(index))
    raw_dir = os.path.join(video_dir, 'raw')

    # Check if "done.txt" exists in the raw directory
    done_file_path = os.path.join(raw_dir, "done.txt")
    if os.path.exists(done_file_path):
        print(f"Skipping download for {index} as 'done.txt' exists.")
        return
    else:
        # Remove the contents of the raw folder if it exists
        if os.path.exists(raw_dir):
            for file in os.listdir(raw_dir):
                file_path = os.path.join(raw_dir, file)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        os.rmdir(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")

    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)

    # Set permissions to 777 for the directories
    os.chmod(video_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    os.chmod(raw_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    ydl_opts = {
        'outtmpl': os.path.join(raw_dir, '%(title)s.%(ext)s'),
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en', 'live_chat'],
        'skip_download': False,
        'format': 'bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]',  # Ensure Full HD, MP4, and AAC audio
        'cookiefile': cookies_file,
        'sleep_interval_subtitles': 1,
        'sleep_interval': 3,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        title = info_dict.get('title', None)
        ydl.download([url])
        print(f"Downloaded video and chat for: {index}")

        # Set permissions to 777 for the downloaded files
        video_file = os.path.join(raw_dir, f"{title}.mp4")
        chat_file = os.path.join(raw_dir, f"{title}.live_chat.json")
        subtitle_file = os.path.join(raw_dir, f"{title}.en.vtt")
        
        if title:
            title_new = sanitize_filename(title)

        if os.path.exists(video_file):
            os.chmod(video_file, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            os.rename(video_file, os.path.join(raw_dir, f"{title_new}.mp4"))
            
        if os.path.exists(chat_file):
            os.chmod(chat_file, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            os.rename(chat_file, os.path.join(raw_dir, f"{title_new}.live_chat.json"))

        if os.path.exists(subtitle_file):
            os.chmod(subtitle_file, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            os.rename(subtitle_file, os.path.join(raw_dir, f"{title_new}.en.vtt"))

        # Check if all files have been renamed and create an empty "done" file
        renamed_video_file = os.path.join(raw_dir, f"{title_new}.mp4")
        renamed_chat_file = os.path.join(raw_dir, f"{title_new}.live_chat.json")
        renamed_subtitle_file = os.path.join(raw_dir, f"{title_new}.en.vtt")

        if os.path.exists(renamed_video_file) and os.path.exists(renamed_chat_file) and os.path.exists(renamed_subtitle_file):
            done_file_path = os.path.join(raw_dir, "done.txt")
            open(done_file_path, 'w').close()  # Create an empty file

def download_video_info(url, output_path, index, cookies_file):
    """
    Download the info JSON for a YouTube video and save the video link in 'done.txt'.

    Parameters:
    url (str): The URL of the YouTube video.
    output_path (str): The base directory where the info JSON will be saved.
    index (int): An index used to create a unique directory for each video.
    cookies_file (str): Path to the cookies file for authentication.

    Returns:
    None
    """
    # Create a directory for each video using an integer index
    video_dir = os.path.join(output_path, str(index))
    raw_dir = os.path.join(video_dir, 'raw')

    # Check if "done.txt" exists in the raw directory
    done_file_path = os.path.join(raw_dir, "done.txt")
    if not os.path.exists(done_file_path):
        print(f"Skipping download for {index} as 'done.txt' does not exist.")
        return

    ydl_opts = {
        'outtmpl': os.path.join(raw_dir, '%(title)s'),
        'skip_download': True,
        'writeinfojson': True,
        'cookiefile': cookies_file,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        title = info_dict.get('title', None)
        ydl.download([url])
        print(f"Downloaded info JSON for: {index}")

        if title:
            title_new = sanitize_filename(title)
            original_info_file = os.path.join(raw_dir, f"{title}.info.json")
            sanitized_info_file = os.path.join(raw_dir, f"{title_new}.info.json")
            
            if os.path.exists(original_info_file):
                os.rename(original_info_file, sanitized_info_file)
                os.chmod(sanitized_info_file, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                
        # Write the URL to the done file
        with open(done_file_path, 'w') as done_file:
            done_file.write(url + '\n')
        
        os.chmod(done_file_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

def main():
    """
    Main function to parse command-line arguments and initiate the download process.

    Returns:
    None
    """
    parser = argparse.ArgumentParser(description='Download YouTube videos and chat history.')
    parser.add_argument('input_file', type=str, help='Path to the text file containing video URLs')
    parser.add_argument('--cookies_file', type=str, default='./cookies.txt', help='Path to the cookies file for authentication (optional)')
    parser.add_argument('--output_path', type=str, default='/mnt-persist/data', help='Output path for downloaded files (optional)')
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    os.chmod(args.output_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    
    with open(args.input_file, 'r') as file:
        urls = file.readlines()

    for index, url in enumerate(urls, start=1):
        url = url.strip()
        if url:
            try:
                #download_video_and_chat(url, args.output_path, index, args.cookies_file)
                download_video_info(url, args.output_path, index, args.cookies_file)
                time.sleep(3)
            except Exception as e:
                print(f"Failed to process {e}: {url}")

    # with open(args.input_file, 'r') as file:
    #     urls = [url.strip() for url in file.readlines() if url.strip()]

    # # Use multiprocessing to download videos concurrently
    # with multiprocessing.Pool(3) as pool:  # Reduce the number of concurrent processes
    #     pool.starmap(download_video_and_chat, [(url, args.output_path, index, args.cookies_file) for index, url in enumerate(urls, start=1)])

if __name__ == "__main__":
    main()
