import gradio as gr
import PIL.Image as Image
import subprocess
import os
import sys
import cv2
import numpy as np

from get_percent import *

def install_dependencies():
    """Install missing dependencies."""
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'IPython'], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to install dependencies: {e}")

def predict_image(file, conf_threshold, iou_threshold, model_type):
    """Predicts objects in an image using a custom YOLOv9 model or TensorRT engine with adjustable confidence and IOU thresholds."""

    output_image_path = "runs/detect/gradio/output.jpg"

    if model_type == "PyTorch":
        weights_path = './runs/train/yolov9_fine/weights/best.pt'
    else:
        weights_path = './runs/train/yolov9_fine/weights/best.engine'
    depth_weights_path = './lite-8m'  # 깊이 예측 모델 가중치 경로
    
    # Check if weights file exists
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights file {weights_path} not found. Ensure the model weights are in the correct directory.")

    try:
        fire_count, smoke_count, fire_max, smoke_max, fire_warning, smoke_warning = run(
            source=file,
            weights=weights_path,
            depth_weights=depth_weights_path,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            name="gradio",
            exist_ok=True,
        )
                
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Inference failed: {e}")

    warning_info = (f"# of Detected fires: {fire_count}, Maximum Severity info for fire: {fire_max:.2f}%, {fire_warning}\n"
                    f"# of Detected smokes: {smoke_count}, Maximum Severity info for smoke: {smoke_max:.2f}%, {smoke_warning}")

    if file.endswith(".jpg"):
        if os.path.exists(output_image_path):
            output_image = Image.open(output_image_path)
        else:
            raise FileNotFoundError(f"Output image not found at {output_image_path}")
        return output_image, warning_info
    else:
        raise ValueError("Unsupported file type. Only JPG images are supported.")
    
iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.File(type="filepath", label="Upload an image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
        gr.Radio(choices=["PyTorch", "TensorRT"], label="Model Type", value="PyTorch")
    ],
    outputs=[gr.Image(type="pil", label="Result"), gr.Textbox(label="Warning Information")],
    title="CV1 project",
    description="Upload images for inference using a custom YOLOv9 model or TensorRT engine.",
)

if __name__ == "__main__":
    iface.launch()
