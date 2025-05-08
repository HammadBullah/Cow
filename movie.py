import ffmpeg
import os

def compress_video(input_path, output_path, target_resolution=(1280, 720), bitrate="1000k"):
    """
    Compress a video using FFmpeg by reducing resolution and bitrate.
    
    Args:
        input_path (str): Path to input video file
        output_path (str): Path to save compressed video
        target_resolution (tuple): Desired resolution (width, height)
        bitrate (str): Target bitrate (e.g., '1000k' for 1000 kbps)
    """
    try:
        # Ensure FFmpeg can find the input file
        stream = ffmpeg.input(input_path)
        
        # Configure compression settings
        stream = ffmpeg.output(
            stream,
            output_path,
            vcodec="libx264",  # H.264 codec for compatibility
            acodec="aac",      # AAC audio codec
            video_bitrate=bitrate,
            s=f"{target_resolution[0]}x{target_resolution[1]}",  # Set resolution
            preset="medium",   # Balance between speed and compression
            threads=4          # Use multiple CPU cores
        )
        
        # Run FFmpeg command
        ffmpeg.run(stream, overwrite_output=True)
        
        print(f"Video compressed successfully! Saved to {output_path}")
        print(f"Original size: {os.path.getsize(input_path) / (1024*1024):.2f} MB")
        print(f"Compressed size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        
    except ffmpeg.Error as e:
        print(f"Error during compression: {e.stderr.decode()}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    # Example usage
    input_video = "/Users/hammadsafi/Downloads/FinalPresentation.MOV"  # Input video path
    output_video = "compressed_video.mp4"  # Output file name
    
    if os.path.exists(input_video):
        compress_video(input_video, output_video)
    else:
        print("Input video file not found!")