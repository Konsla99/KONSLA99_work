import gradio as gr
import PIL.Image as Image
import subprocess
import os
import sys

from models.common import DetectMultiBackend

def install_dependencies():
    """Install missing dependencies."""
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'IPython'], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to install dependencies: {e}")

def predict_image(img, conf_threshold, iou_threshold, model_type):
    """Predicts objects in an image using a custom YOLOv9 model or TensorRT engine with adjustable confidence and IOU thresholds."""
    input_image_path = "input_image.jpg"
    output_image_path = "runs/detect/trt_test/input_image.jpg"
    img.save(input_image_path)

    # print("\nflag\n")

    if model_type == "PyTorch":
        weights_path = './runs/train/yolov9_fine/weights/best.pt'
    else:
        weights_path = './runs/train/yolov9_fine/weights/best.engine'

    detect_script = "detectdepth.py"
    depth_weights_path = './lite-8m'  # 깊이 예측 모델 가중치 경로
    
    # Check if depth_yolo_detect.py script exists
    if not os.path.isfile(detect_script):
        raise FileNotFoundError(f"{detect_script} not found. Ensure the script is in the correct directory.")

    # Check if weights file exists
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights file {weights_path} not found. Ensure the model weights are in the correct directory.")

    # Check if depth weights file exists
    encoder_path = os.path.join(depth_weights_path, "encoder.pth")
    decoder_path = os.path.join(depth_weights_path, "depth.pth")
    if not os.path.isfile(encoder_path) or not os.path.isfile(decoder_path):
        raise FileNotFoundError(f"Depth weights files not found in {depth_weights_path}. Ensure the depth model weights are in the correct directory.")

    # Install missing dependencies
    # install_dependencies()

    # Run the detection script
    command = [
        sys.executable, detect_script,
        "--source", input_image_path,
        "--weights", weights_path,
        "--depth-weights", depth_weights_path,  # 깊이 예측 모델 가중치 경로 추가
        "--conf-thres", str(conf_threshold),
        "--iou-thres", str(iou_threshold),
        "--project", "runs/detect",
        "--name", "trt_test",
        "--exist-ok"
        # "--device", "cpu",
    ]

    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode()
        print(output)
        print(result.stderr.decode())
        
        # Extract depth information from the output
        depth_info = ""
        for line in output.split("\n"):
            if "Depth at" in line:
                depth_info += line + "\n"
                
    except subprocess.CalledProcessError as e:
        error_message = f"Error occurred while running {detect_script}: {e}\n"
        error_message += e.stdout.decode() + "\n"
        error_message += e.stderr.decode()
        raise RuntimeError(error_message)

    # Load the output image
    if os.path.exists(output_image_path):
        output_image = Image.open(output_image_path)
    else:
        raise FileNotFoundError(f"Output image not found at {output_image_path}")

    return output_image, depth_info

iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
        gr.Radio(choices=["PyTorch", "TensorRT"], label="Model Type", value="PyTorch")
    ],
    outputs=[gr.Image(type="pil", label="Result"), gr.Textbox(label="Depth Information")],
    title="Custom YOLOv9 Gradio",
    description="Upload images for inference using a custom YOLOv9 model or TensorRT engine.",
    # live = True,
    # analytics_enabled = True,
)

if __name__ == "__main__":
    iface.launch()
