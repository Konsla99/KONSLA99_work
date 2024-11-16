import os
import subprocess
import argparse

def extract_frames(video_path, input_dir, fps=10):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created input directory: {input_dir}")
    
    output_pattern = os.path.join(input_dir, "%04d.jpg")
    ffmpeg_command = [
        "ffmpeg", "-i", video_path, "-qscale:v", "1", "-qmin", "1",
        "-vf", f"fps={fps}", output_pattern
    ]
    print(f"Running ffmpeg to extract frames to {input_dir}...")
    subprocess.run(ffmpeg_command, check=True)
    print(f"Frames extracted to {input_dir}")

def run_convert_script(root_folder, mydata_name):
    convert_script = os.path.join(root_folder, "convert.py")
    if not os.path.exists(convert_script):
        raise FileNotFoundError(f"convert.py script not found in {root_folder}")

    mydata_path = os.path.join(root_folder, "proj", "custom_data", mydata_name)

    # Run the convert.py script
    command = ["python", convert_script, "-s", mydata_path]
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)

    # Print debug output for convert.py
    print("----- convert.py Output -----")
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)
    print("-----------------------------")

    if result.returncode != 0:
        print(f"convert.py failed with exit code {result.returncode}")
        raise subprocess.CalledProcessError(result.returncode, command)
    print("convert.py completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="COLMAP preparation script")
    parser.add_argument("--originmp4", required=True, help="Name of the input video file (e.g., 's_free').")
    parser.add_argument("--root", default=r"D:/git_hub_repository/Konsla99_work/efficientgaussian", help="Root folder of the project.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second to extract (default: 10).")
    args = parser.parse_args()

    root_folder = args.root
    mydata_name = os.path.splitext(args.originmp4)[0]  # Extract name without extension
    mydata_folder = os.path.join("custom_data", mydata_name)
    mydata_full_path = os.path.join(root_folder, "proj", mydata_folder)
    input_dir = os.path.join(mydata_full_path, "input")
    video_path = os.path.join(mydata_full_path, f"{mydata_name}.mp4")

    print(f"Resolved video path: {video_path}")
    print(f"Dataset folder path: {mydata_full_path}")
    print(f"Input folder path: {input_dir}")

    if not os.path.exists(mydata_full_path):
        os.makedirs(mydata_full_path)
        print(f"Created folder for dataset: {mydata_full_path}")
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created input folder: {input_dir}")

    extract_frames(video_path, input_dir, fps=args.fps)
    run_convert_script(root_folder, mydata_name)

if __name__ == "__main__":
    main()
