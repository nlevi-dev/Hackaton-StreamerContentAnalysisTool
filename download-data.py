import yt_dlp
import os
import argparse
import stat

def download_video_and_chat(url, output_path, index, cookies_file):
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
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',  # Ensure MP4 format
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

        if os.path.exists(video_file):
            os.chmod(video_file, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        if os.path.exists(chat_file):
            os.chmod(chat_file, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

def main():
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
