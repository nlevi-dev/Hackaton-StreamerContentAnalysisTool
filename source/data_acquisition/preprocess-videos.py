import os
import subprocess
import argparse
import stat
import multiprocessing

# Function to process video files
def process_videos(base_dir, video_resolution, video_fps):
    """
    Processes video files in the specified base directory by extracting images and audio.

    This function performs the following operations on video files found within the specified
    base directory:
    1. Extracts images from video files at a specified resolution and frame rate.
    2. Extracts audio tracks and converts them to MP3 format.
    3. Sets appropriate permissions for the extracted files to ensure accessibility.

    Args:
        base_dir (str): The base directory containing subdirectories with video files.
        video_resolution (str): The resolution for the extracted images in the format 'WxH'.
        video_fps (str): The frame rate for image extraction, specified as frames per second.

    The function iterates over each subdirectory within the base directory, processes each video
    file by extracting images and audio, and saves them in designated directories. It also ensures
    that the extracted files have the correct permissions set for user, group, and others.

    Example:
        >>> process_videos('/path/to/base_dir', '1920x1080', '30')
        This will process all video files in '/path/to/base_dir', extracting images at 1920x1080
        resolution and 30 frames per second, and converting audio to MP3 format.
    """
    audio_files = []
    for video_dir in [f.path for f in os.scandir(base_dir) if f.is_dir()]:
        raw_dir = os.path.join(video_dir, 'raw')
        images_dir = os.path.join(video_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        os.chmod(images_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        
        for file in os.listdir(raw_dir):
            if file.endswith(('.mp4', '.avi', '.mov')):  # Add more video formats if needed
                video_path = os.path.join(raw_dir, file)
                audio_mp3_path = os.path.join(video_dir, f'{os.path.splitext(file)[0]}.mp3')
                image_pattern = os.path.join(images_dir, f'{os.path.splitext(file)[0]}_%06d.jpg')

                print(f"Processing video: {video_path}")

                audio_files.append((video_path, audio_mp3_path))
                
                print(f"Extracting images to: {images_dir}")
                subprocess.run(['ffmpeg', '-i', video_path, '-vf', f'fps={video_fps}, scale={video_resolution}, format=yuv420p', '-threads', '16', '-thread_type', 'slice', image_pattern], check=True)

                # Rename images to reflect when the picture was taken from the video
                for image_file in os.listdir(images_dir):
                    if image_file.endswith('.jpg'):
                        image_path = os.path.join(images_dir, image_file)
                        # Extract the timestamp from the image filename
                        frame_number = int(image_file.split('_')[-1].split('.')[0])
                        # Calculate the timestamp in seconds
                        timestamp = int(frame_number / float(video_fps)) - 30 #there is an offset :((
                        # Format the timestamp as seconds
                        timestamp_formatted = f"{timestamp:06}"
                        # Create the new filename with the timestamp
                        new_image_file = f"{timestamp_formatted}_{os.path.splitext(file)[0]}.jpg"
                        new_image_path = os.path.join(images_dir, new_image_file)
                        # Rename the image file
                        os.rename(image_path, new_image_path)

                # Set permissions for all images recursively
                for root, dirs, files in os.walk(images_dir):
                    for image_file in files:
                        image_path = os.path.join(root, image_file)
                        os.chmod(image_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                
                print(f"Set permissions for images: {images_dir}")
        
    # Use multiprocessing to process audio files in parallel
    with multiprocessing.Pool(8) as pool:
        pool.map(process_audio, audio_files)

def process_audio(inputs):
    """
    Extract and process audio from video files.

    This function uses FFmpeg to extract audio tracks from video files and convert them into MP3 format.
    It ensures that the audio files are saved with appropriate permissions for further use.

    Args:
        inputs (tuple): A tuple containing:
            audio_in (str): The file path to the input video file from which audio is to be extracted.
            audio_out (str): The file path where the extracted audio in MP3 format will be saved.

    Steps:
        1. Extracts the audio from the specified video file.
        2. Converts the extracted audio to MP3 format with a sample rate of 44100 Hz and mono channel.
        3. Sets the file permissions to allow read, write, and execute for the user, group, and others.

    Example:
        >>> process_audio(('/path/to/video.mp4', '/path/to/output.mp3'))
        This will extract the audio from 'video.mp4' and save it as 'output.mp3'.
    """
    audio_in, audio_out = inputs
    print(f"Processing audio: {audio_out}")
    subprocess.run(['ffmpeg', '-i', audio_in, '-q:a', '0', '-map', 'a', '-ar', '44100', '-ac', '1', '-threads', '1', audio_out], check=True)
    os.chmod(audio_out, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process videos to extract audio and images.')
    parser.add_argument('base_dir', nargs='?', default='/mnt-persist/data', type=str, help='The base directory containing the data folders.')
    parser.add_argument('--video_resolution', type=str, default='1920x1080', help='Video resolution in WxH format.')
    parser.add_argument('--video_fps', type=str, default='0.016666666', help='Video fps.')
    args = parser.parse_args()

    print(f"Starting processing in directory: {args.base_dir}")
    process_videos(args.base_dir, args.video_resolution, args.video_fps)
    print("Processing complete.") 