import cv2
import numpy as np
import argparse
import os

def apply_chroma_key(video_path, background_path, output_path="output_video.mp4"):
    """
    Apply chroma key effect to a video with a green screen background.

    Args:
    - video_path: Path to the input video with green screen.
    - background_path: Path to the background image to use.
    - output_path: Path where the output video will be saved.
    """
    # Load background image
    background = cv2.imread(background_path)
    if background is None:
        raise ValueError("Background image could not be loaded. Check the path.")

    # Open video file
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise ValueError("Video file could not be opened. Check the path.")
    
    # Get video properties
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30  # Default FPS if unable to fetch
    
    # Resize background to match video frame size
    bg_height, bg_width = background.shape[:2]
    if (bg_width != frame_width) or (bg_height != frame_height):
        print(f"Resizing background from ({bg_width}, {bg_height}) to ({frame_width}, {frame_height})")
        background = cv2.resize(background, (frame_width, frame_height))
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Alternative codec
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Convert frame to HSV color space for better color range filtering
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define the range for green color in HSV
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        
        # Create mask to detect green areas
        mask = cv2.inRange(hsv_frame, lower_green, upper_green)
        mask_inv = cv2.bitwise_not(mask)
        
        # Extract the part of the frame without the green background
        foreground = cv2.bitwise_and(frame, frame, mask=mask_inv)
        background_part = cv2.bitwise_and(background, background, mask=mask)
        
        # Combine the foreground and the background
        combined_frame = cv2.add(foreground, background_part)
        
        # Write the frame to the output video
        video_writer.write(combined_frame)

    # Release resources
    video_capture.release()
    video_writer.release()
    print("Video processing completed. Output saved to:", output_path)

if __name__ == "__main__":
    # Default base paths
    default_video_base = r"D:\\git_hub_repository\\KONSLA99_work\\efficientgaussian\\proj\\green_screen"
    default_background_base = r"D:\\git_hub_repository\\KONSLA99_work\\efficientgaussian\\proj\\background"
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Apply Chroma Key effect to a video.")
    parser.add_argument("--inv", required=True, help="Filename of the input video (e.g., 'video.mp4').")
    parser.add_argument("--bg", required=True, help="Filename of the background image (e.g., 'background.jpg').")
    parser.add_argument("--o", default="output_video.mp4", help="Path to save the output video.")
    
    args = parser.parse_args()

    # Construct full paths
    video_path = os.path.join(default_video_base, args.inv)
    background_path = os.path.join(default_background_base, args.bg)

    # Call the function with constructed paths
    apply_chroma_key(video_path, background_path, args.o)
