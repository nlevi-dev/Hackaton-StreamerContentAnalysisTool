import os
import subprocess
import argparse
import stat
import multiprocessing

# Function to process video files
def process_videos(base_dir, video_resolution, video_fps):
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
                subprocess.run(['ffmpeg', '-i', video_path, '-vf', f'fps={video_fps}, scale={video_resolution}, format=yuv420p', '-threads', '20', '-thread_type', 'slice', image_pattern], check=True)

                # Set permissions for all images recursively
                for root, dirs, files in os.walk(images_dir):
                    for image_file in files:
                        image_path = os.path.join(root, image_file)
                        os.chmod(image_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                
                print(f"Set permissions for images: {images_dir}")
        
    # Use multiprocessing to process audio files in parallel
    with multiprocessing.Pool(multiprocessing.cpu_count()-8) as pool:
        pool.map(process_audio, audio_files)

def process_audio(inputs):
    audio_in, audio_out = inputs
    print(f"Processing audio: {audio_out}")
    subprocess.run(['ffmpeg', '-i', audio_in, '-q:a', '0', '-map', 'a', '-ar', '44100', '-ac', '1', '-threads', '1', audio_out], check=True)
    os.chmod(audio_out, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process videos to extract audio and images.')
    parser.add_argument('base_dir', nargs='?', default='/mnt-persist/data', type=str, help='The base directory containing the data folders.')
    parser.add_argument('--video_resolution', type=str, default='1920x1080', help='Video resolution in WxH format.')
    parser.add_argument('--video_fps', type=str, default='1/5', help='Video fps.')
    args = parser.parse_args()

    print(f"Starting processing in directory: {args.base_dir}")
    process_videos(args.base_dir, args.video_resolution, args.video_fps)
    print("Processing complete.") 