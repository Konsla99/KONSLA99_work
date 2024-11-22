import gradio as gr
import cv2
import subprocess
import os
import sys
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from moviepy.editor import ImageSequenceClip

def process_frame(frame, frame_idx, conf_threshold, iou_threshold, model_type, weights_path, depth_weights_path):
    frame_path = f"current_frame_{frame_idx}.jpg"
    cv2.imwrite(frame_path, frame)
    
    command = [
        sys.executable, "detectdepth.py",
        "--source", frame_path,
        "--weights", weights_path,
        "--depth-weights", depth_weights_path,
        "--conf-thres", str(conf_threshold),
        "--iou-thres", str(iou_threshold),
        "--project", "runs/detect",
        "--name", "trt_test",
        "--exist-ok"
    ]
    
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode()
        print(output)
        print(result.stderr.decode())
        
        processed_frame_path = "runs/detect/trt_test/current_frame.jpg"
        if os.path.exists(processed_frame_path):
            processed_frame = cv2.imread(processed_frame_path)
            return processed_frame
        else:
            raise FileNotFoundError(f"Processed frame not found at {processed_frame_path}")
            
    except subprocess.CalledProcessError as e:
        error_message = f"Error occurred while running detectdepth.py: {e}\n"
        error_message += e.stdout.decode() + "\n"
        error_message += e.stderr.decode()
        raise RuntimeError(error_message)

def predict_video(video_path, conf_threshold, iou_threshold, model_type):
    start_time = time.time()
    
    input_video_path = video_path  # video_path를 직접 사용
    output_video_path = "runs/detect/trt_test/output_video.mp4"

    video_save_time = time.time()
    print(f"Video save time: {video_save_time - start_time} seconds")

    if model_type == "PyTorch":
        weights_path = './runs/train/yolov9_fine/weights/best.pt'
    else:
        weights_path = './runs/train/yolov9_fine/weights/best.engine'

    depth_weights_path = './lite-8m'
    encoder_path = os.path.join(depth_weights_path, "encoder.pth")
    decoder_path = os.path.join(depth_weights_path, "depth.pth")
    
    script_setup_time = time.time()
    print(f"Script setup time: {script_setup_time - video_save_time} seconds")

    # Extract frames from the video
    cap = cv2.VideoCapture(input_video_path)
    frames = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append((frame, frame_idx))
        frame_idx += 1
    cap.release()
    
    frame_extraction_time = time.time()
    print(f"Frame extraction time: {frame_extraction_time - script_setup_time} seconds")

    # Process each frame with the YOLO model in parallel
    processed_frames = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_frame, frame, idx, conf_threshold, iou_threshold, model_type, weights_path, depth_weights_path) for frame, idx in frames]
        for future in futures:
            processed_frames.append(future.result())

    processing_time = time.time()
    print(f"Frame processing time: {processing_time - frame_extraction_time} seconds")

    # Save processed frames as video
    clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in processed_frames], fps=24)
    clip.write_videofile(output_video_path, codec='libx264')

    video_save_time = time.time()
    print(f"Processed video save time: {video_save_time - processing_time} seconds")

    total_time = video_save_time - start_time
    print(f"Total time: {total_time} seconds")

    return output_video_path

iface = gr.Interface(
    fn=predict_video,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
        gr.Radio(choices=["PyTorch", "TensorRT"], label="Model Type", value="PyTorch")
    ],
    outputs=gr.Video(label="Result Video"),
    title="Custom YOLOv9 Gradio",
    description="Upload videos for inference using a custom YOLOv9 model or TensorRT engine.",
)

if __name__ == "__main__":
    iface.launch()
