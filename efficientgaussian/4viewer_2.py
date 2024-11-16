import sys
import time
import subprocess
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog, QGridLayout
)
import win32gui
import win32con
import ctypes


class SIBRViewerApp(QMainWindow):
    def __init__(self, sibr_path):
        super().__init__()
        self.setWindowTitle("SIBR Viewer Embedded - 2x2 Layout")
        self.setGeometry(100, 100, 1600, 900)  # Adjusted Y-axis height for the main window

        self.sibr_path = sibr_path
        self.viewers = []  # Store viewer instances

        # Main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        layout = QGridLayout(self.main_widget)

        # Create 4 viewers in a 2x2 layout
        for row in range(2):
            for col in range(2):
                viewer = SIBRViewer(self.sibr_path, row, col)
                self.viewers.append(viewer)
                layout.addWidget(viewer, row, col)



class SIBRViewer(QWidget):
    detected_windows = []  # Class-level variable to store detected window handles

    def __init__(self, sibr_path, row, col):
        super().__init__()
        self.sibr_path = sibr_path
        self.sibr_process = None
        self.sibr_window = None
        self.row = row
        self.col = col

        # Main layout
        layout = QVBoxLayout(self)

        # Folder selection button
        self.select_button = QPushButton(f"Select Folder for Viewer ({row}, {col})", self)
        self.select_button.clicked.connect(self.select_folder)
        layout.addWidget(self.select_button)

        # Container for the embedded window
        self.viewer_container = QWidget(self)
        self.viewer_container.setStyleSheet("background-color: black;")
        self.viewer_container.setMinimumSize(500, 300)  # Adjusted size for smaller viewers
        layout.addWidget(self.viewer_container)

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, f"Select Folder for Viewer ({self.row}, {self.col})")
        if folder_path:
            print(f"Selected folder for Viewer ({self.row}, {self.col}): {folder_path}")
            self.run_sibr_viewer(folder_path)

    def run_sibr_viewer(self, folder_path):
        if self.sibr_process is not None:
            print(f"SIBR Viewer ({self.row}, {self.col}) is already running. Restarting with new folder...")
            self.sibr_process.terminate()
            self.sibr_process = None

        # Start the SIBR viewer as a subprocess
        self.sibr_process = subprocess.Popen(
            [self.sibr_path, "-m", folder_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"SIBR Viewer ({self.row}, {self.col}) started.")
        time.sleep(2)  # Wait for the window to open

        # Find the SIBR viewer window handle
        self.sibr_window = self.find_sibr_window()
        if self.sibr_window:
            print(f"Found SIBR window for Viewer ({self.row}, {self.col}): {self.sibr_window}")
            self.embed_sibr_window()
        else:
            print(f"Failed to find SIBR Viewer window for Viewer ({self.row}, {self.col}).")

    def find_sibr_window(self):
        def callback(hwnd, extra):
            """Callback to enumerate all windows and append to the list if it matches."""
            window_text = win32gui.GetWindowText(hwnd)
            class_name = win32gui.GetClassName(hwnd)
            if "sibr" in window_text.lower() or "sibr" in class_name.lower():
                if hwnd not in SIBRViewer.detected_windows:  # Avoid duplicates
                    extra.append(hwnd)

        hwnds = []
        win32gui.EnumWindows(callback, hwnds)

        if len(hwnds) > 0:
            hwnd = hwnds[0]
            SIBRViewer.detected_windows.append(hwnd)  # Cache the detected window
            print(f"Assigning Viewer ({self.row}, {self.col}) to HWND: {hwnd}")
            return hwnd
        else:
            print(f"No matching window found for Viewer ({self.row}, {self.col})")
            return None

    def embed_sibr_window(self):
        if not self.sibr_window:
            print(f"[ERROR] SIBR window handle for Viewer ({self.row}, {self.col}) is invalid. Retrying...")
            self.sibr_window = self.find_sibr_window()
            if not self.sibr_window:
                print(f"[ERROR] Failed to find SIBR window for Viewer ({self.row}, {self.col}) after retry.")
                return

        # Get the HWND of the specific viewer's container
        container_hwnd = int(self.viewer_container.winId())
        print(f"Embedding SIBR window for Viewer ({self.row}, {self.col}): {self.sibr_window} into container: {container_hwnd}")

        try:
            # Change window style to make it embeddable
            win32gui.SetParent(self.sibr_window, container_hwnd)
            win32gui.SetWindowLong(
                self.sibr_window,
                win32con.GWL_STYLE,
                win32gui.GetWindowLong(self.sibr_window, win32con.GWL_STYLE)
                & ~win32con.WS_CAPTION
                & ~win32con.WS_THICKFRAME,
            )

            # Resize and reposition the embedded window
            self.adjust_embedded_window()
        except Exception as e:
            print(f"Failed to embed SIBR window for Viewer ({self.row}, {self.col}): {e}")

    def adjust_embedded_window(self):
        """Adjust the size and position of the embedded SIBR window."""
        win32gui.SetWindowPos(
            self.sibr_window,
            None,
            0,  # X position
            0,  # Y position
            self.viewer_container.width(),
            self.viewer_container.height(),
            win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE,
        )

    def resizeEvent(self, event):
        """Handle resizing of the main window."""
        super().resizeEvent(event)
        if self.sibr_window:
            self.adjust_embedded_window()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Absolute path to the SIBR viewer executable
    sibr_viewer_path = "D:/code/3dgs/viewers/bin/SIBR_GaussianViewer_app.exe"

    viewer_app = SIBRViewerApp(sibr_viewer_path)
    viewer_app.show()

    sys.exit(app.exec_())
