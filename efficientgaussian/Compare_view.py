import sys
import time
import subprocess
import shutil
import os
import json
import re
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFileDialog, QGridLayout, QLabel
)
import win32gui
import win32con
import ctypes


class SIBRViewerApp(QMainWindow):
    def __init__(self, sibr_path):
        super().__init__()
        self.setWindowTitle("SIBR Viewer with Metrics")
        self.setGeometry(100, 100, 1800, 1000)

        self.sibr_path = sibr_path
        self.viewers = []

        # 매인 위제 및 레이아웃 설정
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        main_layout = QVBoxLayout(self.main_widget)

        # 상단 버튼 레이아웃 (모델 선택, 학습 시작, 렌더 확인 버튼)
        button_layout = QHBoxLayout()
        self.dataset_button = QPushButton("모델 선택", self)
        self.dataset_button.clicked.connect(self.select_dataset)
        button_layout.addWidget(self.dataset_button)

        self.dataset_label = QLabel("선택된 데이터셋 없음")
        button_layout.addWidget(self.dataset_label)

        self.train_button = QPushButton("학습 시작", self)
        self.train_button.clicked.connect(self.train_models)
        button_layout.addWidget(self.train_button)

        main_layout.addLayout(button_layout)

        # 상단의 두 개의 검은 화면 (Improve, Eagles)
        viewer_layout = QHBoxLayout()
        viewer_1_layout = QVBoxLayout()
        viewer_2_layout = QVBoxLayout()

        self.viewer_1_label = QLabel("<b>Improve</b>", self)
        viewer_1_layout.addWidget(self.viewer_1_label)
        self.viewer_1 = SIBRViewer(self.sibr_path, 0, 0)
        viewer_1_layout.addWidget(self.viewer_1)

        self.viewer_2_label = QLabel("<b>EAGLES</b>", self)
        viewer_2_layout.addWidget(self.viewer_2_label)
        self.viewer_2 = SIBRViewer(self.sibr_path, 0, 1)
        viewer_2_layout.addWidget(self.viewer_2)

        viewer_layout.addLayout(viewer_1_layout)
        viewer_layout.addLayout(viewer_2_layout)
        main_layout.addLayout(viewer_layout)

        # 렌더 확인 버튼
        self.render_button = QPushButton("렌더 확인", self)
        self.render_button.clicked.connect(self.render_models)
        main_layout.addWidget(self.render_button)

        # 성능 지표 표시 레이아웃 (3행 2열로 구성)
        metrics_layout = QGridLayout()
        self.metric_labels = {}
        metrics_names = [
            "<b>☐ Metric - Model 1 ☐</b>", "<b>☐ Metric - Model 2 ☐</b>",
            "<b>☐ Time - Model 1 ☐</b>", "<b>☐ Time - Model 2 ☐</b>",
            "<b>☐ Size - Model 1 ☐</b>", "<b>☐ Size - Model 2 ☐</b>"
        ]

        for i, name in enumerate(metrics_names):
            row, col = divmod(i, 2)
            label = QLabel(name, self)
            value_label = QLabel("", self)
            value_label.setStyleSheet("background-color: white; border: 1px solid black;")
            value_label.setAlignment(Qt.AlignCenter)
            self.metric_labels[name] = value_label
            metrics_layout.addWidget(label, row * 2, col)
            metrics_layout.addWidget(value_label, row * 2 + 1, col)

        main_layout.addLayout(metrics_layout)

    def select_dataset(self):
        folder_path = QFileDialog.getExistingDirectory(self, "데이터셋 선택")
        if folder_path:
            self.dataset_label.setText(f"선택된 데이터셋: {folder_path.split('/')[-1]}")
            self.dataset_path = folder_path
            self.lowest_dir_name = folder_path.split('/')[-1]

    def train_models(self):
        if hasattr(self, 'dataset_path'):
            log_path_1 = "./log_dir/final/train_log_model_1.txt"
            log_path_2 = "./log_dir/final/train_log_model_2.txt"

            # 1단계: d_train.py로 첫 번째 모델 학습 (로그 저장)
            with open(log_path_1, "w") as log_file:
                subprocess.run([
                    "python", "./d_train.py", "--config", "./configs/i-eagles.yaml",
                    "-s", self.dataset_path,
                    "-m", f"./log_dir/final/{self.lowest_dir_name}_model_1", "--save_ply"
                ], stdout=log_file, stderr=log_file)

            # 2단계: train_eval.py로 두 번째 모델 학습 (로그 저장)
            with open(log_path_2, "w") as log_file:
                subprocess.run([
                    "python", "./train_eval.py", "--config", "./configs/efficient-3dgs.yaml",
                    "-s", self.dataset_path,
                    "-m", f"./log_dir/final/{self.lowest_dir_name}_model_2", "--save_ply"
                ], stdout=log_file, stderr=log_file)

            # 학습 후 지표 업데이트
            self.update_metrics()

            # PLY 및 PKL 파일 이동
            self.move_ply_and_pkl_files()
        else:
            self.dataset_label.setText("먼저 데이터셋을 선택하세요.")

    def update_metrics(self):
        try:
            # 첫 번째 모델 성능 지표 추적
            log_path_1 = "./log_dir/final/train_log_model_1.txt"
            with open(log_path_1, "r") as log_file:
                log_content_1 = log_file.read()
                psnr_1 = self.extract_metric(log_content_1, "PSNR")
                ssim_1 = self.extract_metric(log_content_1, "SSIM")
                lpips_1 = self.extract_metric(log_content_1, "LPIPS")
                training_time_1 = self.extract_time(log_content_1, "Training time")
                render_time_1 = self.extract_time(log_content_1, "Rendering time")
                ply_size_1 = self.get_file_size(f"./log_dir/final/{self.lowest_dir_name}_model_1/point_cloud_best/point_cloud.ply")
                pkl_size_1 = self.get_file_size(f"./log_dir/final/{self.lowest_dir_name}_model_1/point_cloud_best/point_cloud_compressed.pkl")

                # 성능 지표 표시 (첫 번째 모델)
                self.metric_labels["<b>☐ Metric - Model 1 ☐</b>"].setText(f"PSNR: {psnr_1}, SSIM: {ssim_1}, LPIPS: {lpips_1}")
                self.metric_labels["<b>☐ Time - Model 1 ☐</b>"].setText(f"Training: {training_time_1} seconds, Rendering: {render_time_1} seconds")
                self.metric_labels["<b>☐ Size - Model 1 ☐</b>"].setText(f"PLY: {ply_size_1:.2f}MB, PKL: {pkl_size_1:.2f}MB")

            # 두 번째 모델 성능 지표 추적
            log_path_2 = "./log_dir/final/train_log_model_2.txt"
            with open(log_path_2, "r") as log_file:
                log_content_2 = log_file.read()
                psnr_2 = self.extract_metric(log_content_2, "PSNR")
                ssim_2 = self.extract_metric(log_content_2, "SSIM")
                lpips_2 = self.extract_metric(log_content_2, "LPIPS")
                training_time_2 = self.extract_time(log_content_2, "Training time")
                render_time_2 = self.extract_time(log_content_2, "Rendering time")
                ply_size_2 = self.get_file_size(f"./log_dir/final/{self.lowest_dir_name}_model_2/point_cloud_best/point_cloud.ply")
                pkl_size_2 = self.get_file_size(f"./log_dir/final/{self.lowest_dir_name}_model_2/point_cloud_best/point_cloud_compressed.pkl")

                # 성능 지표 표시 (두 번째 모델)
                self.metric_labels["<b>☐ Metric - Model 2 ☐</b>"].setText(f"PSNR: {psnr_2}, SSIM: {ssim_2}, LPIPS: {lpips_2}")
                self.metric_labels["<b>☐ Time - Model 2 ☐</b>"].setText(f"Training: {training_time_2} seconds, Rendering: {render_time_2} seconds")
                self.metric_labels["<b>☐ Size - Model 2 ☐</b>"].setText(f"PLY: {ply_size_2:.2f}MB, PKL: {pkl_size_2:.2f}MB")

            # 파일 크기 비교 및 차이 퍼센트 계산
            if ply_size_1 != 0 and ply_size_2 != 0:
                ply_size_percentage = (abs(ply_size_1 - ply_size_2) / max(ply_size_1, ply_size_2)) * 100
                pkl_size_percentage = (abs(pkl_size_1 - pkl_size_2) / max(pkl_size_1, pkl_size_2)) * 100
                training_time_percentage = (abs(training_time_1 - training_time_2) / max(training_time_1, training_time_2)) * 100
                render_time_percentage = (abs(render_time_1 - render_time_2) / max(render_time_1, render_time_2)) * 100

                self.metric_labels["<b>☐ Size - Model 1 ☐</b>"].setText(
                    f"PLY: {ply_size_1:.2f}MB ({ply_size_percentage:.2f}% smaller), PKL: {pkl_size_1:.2f}MB ({pkl_size_percentage:.2f}% smaller)"
                    if ply_size_1 < ply_size_2 else
                    f"PLY: {ply_size_1:.2f}MB, PKL: {pkl_size_1:.2f}MB"
                )
                self.metric_labels["<b>☐ Size - Model 2 ☐</b>"].setText(
                    f"PLY: {ply_size_2:.2f}MB ({ply_size_percentage:.2f}% smaller), PKL: {pkl_size_2:.2f}MB ({pkl_size_percentage:.2f}% smaller)"
                    if ply_size_2 < ply_size_1 else
                    f"PLY: {ply_size_2:.2f}MB, PKL: {pkl_size_2:.2f}MB"
                )

                self.metric_labels["<b>☐ Time - Model 1 ☐</b>"].setText(
                    f"Training: {training_time_1:.2f} seconds ({training_time_percentage:.2f}% smaller), Rendering: {render_time_1:.2f} seconds ({render_time_percentage:.2f}% smaller)"
                    if training_time_1 < training_time_2 else
                    f"Training: {training_time_1:.2f} seconds, Rendering: {render_time_1:.2f} seconds"
                )
                self.metric_labels["<b>☐ Time - Model 2 ☐</b>"].setText(
                    f"Training: {training_time_2:.2f} seconds ({training_time_percentage:.2f}% smaller), Rendering: {render_time_2:.2f} seconds ({render_time_percentage:.2f}% smaller)"
                    if training_time_2 < training_time_1 else
                    f"Training: {training_time_2:.2f} seconds, Rendering: {render_time_2:.2f} seconds"
                )
        except FileNotFoundError:
            self.metric_labels["<b>☐ Metric - Model 1 ☐</b>"].setText("학습 결과 파일을 찾을 수 없습니다.")
            self.metric_labels["<b>☐ Metric - Model 2 ☐</b>"].setText("학습 결과 파일을 찾을 수 없습니다.")

    def extract_metric(self, log_content, metric_name):
        pattern = rf"{metric_name}\s*:\s*([0-9\.]+)"
        match = re.search(pattern, log_content)
        return float(match.group(1)) if match else 0.0

    def extract_time(self, log_content, time_name):
        pattern = rf"{time_name}:\s*([0-9\.]+) seconds"
        match = re.search(pattern, log_content)
        return float(match.group(1)) if match else 0.0

    def get_file_size(self, file_path):
        try:
            return os.path.getsize(file_path) / (1024 * 1024)
        except FileNotFoundError:
            return 0.0

    def move_ply_and_pkl_files(self):
        # 각 모델의 PLY 및 PKL 파일을 이동
        source_ply_1 = f"./log_dir/final/{self.lowest_dir_name}_model_1/point_cloud_best/point_cloud.ply"
        source_pkl_1 = f"./log_dir/final/{self.lowest_dir_name}_model_1/point_cloud_best/point_cloud_compressed.pkl"
        destination_dir_1 = f"./log_dir/final/{self.lowest_dir_name}_model_1/point_cloud/iteration_7000/"

        # iteration_7000에 있는 기존 PKL 파일 삭제
        pkl_to_delete_1 = os.path.join(destination_dir_1, "point_cloud_compressed.pkl")
        if os.path.exists(pkl_to_delete_1):
            os.remove(pkl_to_delete_1)

        shutil.move(source_ply_1, os.path.join(destination_dir_1, "point_cloud.ply"))
        shutil.move(source_pkl_1, os.path.join(destination_dir_1, "point_cloud_compressed.pkl"))

        source_ply_2 = f"./log_dir/final/{self.lowest_dir_name}_model_2/point_cloud_best/point_cloud.ply"
        source_pkl_2 = f"./log_dir/final/{self.lowest_dir_name}_model_2/point_cloud_best/point_cloud_compressed.pkl"
        destination_dir_2 = f"./log_dir/final/{self.lowest_dir_name}_model_2/point_cloud/iteration_7000/"

        # iteration_7000에 있는 기존 PKL 파일 삭제
        pkl_to_delete_2 = os.path.join(destination_dir_2, "point_cloud_compressed.pkl")
        if os.path.exists(pkl_to_delete_2):
            os.remove(pkl_to_delete_2)

        shutil.move(source_ply_2, os.path.join(destination_dir_2, "point_cloud.ply"))
        shutil.move(source_pkl_2, os.path.join(destination_dir_2, "point_cloud_compressed.pkl"))

    def render_models(self):
        if hasattr(self, 'dataset_path'):
            self.viewer_1.run_sibr_viewer(f"./log_dir/final/{self.lowest_dir_name}_model_1")
            time.sleep(2)  # 첫 번째 뷰어가 열리는 시간을 늘랄
            self.viewer_2.run_sibr_viewer(f"./log_dir/final/{self.lowest_dir_name}_model_2")
        else:
            self.dataset_label.setText("먼저 데이터셋을 선택하세요.")


class SIBRViewer(QWidget):
    detected_windows = []  # 클래스 레버 변수로 윈도우 핸들을 저장

    def __init__(self, sibr_path, row, col):
        super().__init__()
        self.sibr_path = sibr_path
        self.sibr_process = None
        self.sibr_window = None
        self.row = row
        self.col = col

        # 매인 레이아웃 설정
        layout = QVBoxLayout(self)

        # 임베디듨 창을 위한 켄테이너
        self.viewer_container = QWidget(self)
        self.viewer_container.setStyleSheet("background-color: black;")
        self.viewer_container.setMinimumSize(500, 300)  # 뷰어 크기 설정
        layout.addWidget(self.viewer_container)

    def run_sibr_viewer(self, folder_path):
        if self.sibr_process is not None:
            print(f"Viewer ({self.row}, {self.col})가 이미 실행 중입니다. 새 폴더로 재시작합니다...")
            self.sibr_process.terminate()
            self.sibr_process.wait()  # 프로세스가 종료될 때까지 기다림
            self.sibr_process = None

        # SIBR 뷰어를 서브프레시로 시작
        self.sibr_process = subprocess.Popen(
            [self.sibr_path, "-m", folder_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"Viewer ({self.row}, {self.col}) 시작됨.")
        time.sleep(2)  # 창이 열리기 까지 대기 시간을 늘랄

        # SIBR 뷰어 창 핸들 찾기
        self.sibr_window = self.find_sibr_window()
        if self.sibr_window:
            print(f"Viewer ({self.row}, {self.col})의 SIBR 창 찾음: {self.sibr_window}")
            self.embed_sibr_window()
        else:
            print(f"Viewer ({self.row}, {self.col})의 SIBR 창을 찾지 못했습니다.")

    def find_sibr_window(self):
        def callback(hwnd, extra):
            """모든 창을 열거하고 일치하는 것을 리스트에 추가합니다.
            """
            window_text = win32gui.GetWindowText(hwnd)
            class_name = win32gui.GetClassName(hwnd)
            if "sibr" in window_text.lower() or "sibr" in class_name.lower():
                if hwnd not in SIBRViewer.detected_windows:  # 중복 방지
                    extra.append(hwnd)

        hwnds = []
        win32gui.EnumWindows(callback, hwnds)

        if len(hwnds) > 0:
            hwnd = hwnds[0]
            SIBRViewer.detected_windows.append(hwnd)  # 검지된 창 케이시
            print(f"Viewer ({self.row}, {self.col})에 HWND 할당: {hwnd}")
            return hwnd
        else:
            print(f"Viewer ({self.row}, {self.col})에 일치하는 창을 찾지 못했습니다.")
            return None

    def embed_sibr_window(self):
        if not self.sibr_window:
            print(f"[ERROR] Viewer ({self.row}, {self.col})의 SIBR 창 핸들이 유형하지 않습니다. 재시도 중...")
            self.sibr_window = self.find_sibr_window()
            if not self.sibr_window:
                print(f"[ERROR] Viewer ({self.row}, {self.col})의 SIBR 창을 찾는 데 실패했습니다.")
                return

        # 특정 뷰어 켄테이너의 HWND 검사기
        container_hwnd = int(self.viewer_container.winId())
        print(f"컨테이너 핸들: {container_hwnd}, 윈도우 핸들: {self.sibr_window}")
        print(f"Viewer ({self.row}, {self.col})의 SIBR 창을 켄테이너에 임베디드: {self.sibr_window} -> {container_hwnd}")

        try:
            # 창 핸들의 유형성 검사
            if not win32gui.IsWindow(self.sibr_window):
                raise Exception("유형하지 않은 창 핸들입니다.")

            # 창 스타일 변경하여 임베디드 가능하도록 설정
            win32gui.SetParent(self.sibr_window, container_hwnd)
            win32gui.SetWindowLong(
                self.sibr_window,
                win32con.GWL_STYLE,
                win32gui.GetWindowLong(self.sibr_window, win32con.GWL_STYLE)
                & ~win32con.WS_CAPTION
                & ~win32con.WS_THICKFRAME
                | win32con.WS_CHILD
            )

            # 임베디된 창 크기 및 위치 조정
            self.adjust_embedded_window()
        except Exception as e:
            print(f"Viewer ({self.row}, {self.col})의 SIBR 창 임베디어 실패: {e}")
            self.find_sibr_window()
            self.retry_embed()

    def retry_embed(self):
        max_retries = 3  # 재시도 횟수 제한
        for attempt in range(max_retries):
            print(f"Viewer ({self.row}, {self.col}) 임베디어 재시도 중... ({attempt + 1}/{max_retries})")
            time.sleep(2)  # 재시도 전 잠시 대기
            try:
                self.embed_sibr_window()
                if self.sibr_window and win32gui.IsWindow(self.sibr_window):
                    return
            except Exception as e:
                print(f"임베디어 재시도 실패: {e}")

        print(f"Viewer ({self.row}, {self.col}) 임베디어에 반복적으로 실패했습니다.")

    def adjust_embedded_window(self):
        """임베디된 SIBR 창의 크기 및 위치 조정.
        """
        if self.sibr_window and win32gui.IsWindow(self.sibr_window):
            try:
                win32gui.SetWindowPos(
                    self.sibr_window,
                    None,
                    0,  # X 위치
                    0,  # Y 위치
                    self.viewer_container.width(),
                    self.viewer_container.height(),
                    win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE,
                )
            except Exception as e:
                print(f"Viewer ({self.row}, {self.col})의 SIBR 창 위치 조정 실패: {e}")

    def resizeEvent(self, event):
        """매인 창 크기 조절 처리.
        """
        super().resizeEvent(event)
        if self.sibr_window and win32gui.IsWindow(self.sibr_window):
            self.adjust_embedded_window()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # SIBR 뷰어 실행 파일의 절득 경로
    sibr_viewer_path = "D:/code/3dgs/viewers/bin/SIBR_GaussianViewer_app.exe"

    viewer_app = SIBRViewerApp(sibr_viewer_path)
    viewer_app.show()

    sys.exit(app.exec_())
