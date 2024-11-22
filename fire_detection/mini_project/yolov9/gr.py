# 비디오 표시하기
import gradio as gr

def video_display(input_img):
    return input_img

demo = gr.Interface(video_display, gr.Video(), "video")
demo.launch()