import yt_dlp
import os
import argparse
import stat
import re

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

    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)

    # Set permissions to 777 for the directories
    os.chmod(video_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    os.chmod(raw_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    ydl_opts = {
        'outtmpl': os.path.join(raw_dir, '%(title)s.%(ext)s'),
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['live_chat'],
        'skip_download': False,
        'format': 'bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]',  # Ensure Full HD, MP4, and AAC audio
        'cookiefile': cookies_file,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        title = info_dict.get('title', None)
        ydl.download([url])
        print(f"Downloaded video and chat for: {url}")

        # Set permissions to 777 for the downloaded files
        video_file = os.path.join(raw_dir, f"{title}.mp4")
        chat_file = os.path.join(raw_dir, f"{title}.live_chat.json")
        
        if title:
            title_new = sanitize_filename(title)

        if os.path.exists(video_file):
            os.chmod(video_file, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            os.rename(video_file, os.path.join(raw_dir, f"{title_new}.mp4"))
            
        if os.path.exists(chat_file):
            os.chmod(chat_file, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            os.rename(chat_file, os.path.join(raw_dir, f"{title_new}.live_chat.json"))

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
                download_video_and_chat(url, args.output_path, index, args.cookies_file)
            except Exception as e:
                print(f"Failed to process {url}: {e}")

if __name__ == "__main__":
    main()
