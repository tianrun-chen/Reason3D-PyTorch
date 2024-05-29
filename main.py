#Author: Tianrun Chen
#Dated: 2024-May-29 KOKONI 3D 
#Moxin (HUZHOU) Technology Co., LTD.

import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QTextEdit, QLineEdit, QLabel, QFileDialog, QListWidget, QListWidgetItem, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtkmodules.all as vtk

import subprocess
import trimesh
import time

OUTPUT_DIR = "output"
STEP_1_DIR = os.path.join(OUTPUT_DIR, "step_1")
STEP_2_DIR = os.path.join(OUTPUT_DIR, "step_2")
STEP_3_DIR = os.path.join(OUTPUT_DIR, "step_3")

POSE_FILE = "data/pose.txt"

class EmittingStream(object):
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.append(message)

    def flush(self):
        pass

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.view_check("front")

    def initUI(self):
        self.setWindowTitle("3D Reasoning Part Segmentation with LLM")
        self.setGeometry(100, 100, 1600, 600)

        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        layout = QHBoxLayout(centralWidget)

        left_panel = QVBoxLayout()
        layout.addLayout(left_panel)

        self.initImageDisplayArea(left_panel)

        mid_panel = QVBoxLayout()
        layout.addLayout(mid_panel)

        # 3D model display area
        self.init3DDisplayArea(mid_panel)

        # Right panel layout
        right_panel = QVBoxLayout()
        layout.addLayout(right_panel)

        # File path input area
        self.initFilePathArea(right_panel)

        # Text prompt input area
        self.initTextPromptArea(right_panel)

        # Text prompt input area
        self.initBottonArea(right_panel)

        # Log display area
        self.initLogDisplayArea(right_panel)

    def initImageDisplayArea(self, layout):
        self.image_display_area = QLabel()
        self.image_display_area.setFixedSize(600, 600)  # Set the size to 400x400 pixels
        self.image_display_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_display_area.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_display_area)

        # Control buttons
        btn_layout = QHBoxLayout()

        self.btnViewL = QPushButton("Previous", self)
        self.btnViewL.clicked.connect(self.show_previous_image)
        btn_layout.addWidget(self.btnViewL)

        self.btnViewF = QPushButton("Next", self)
        self.btnViewF.clicked.connect(self.show_next_image)
        btn_layout.addWidget(self.btnViewF)

        layout.addLayout(btn_layout)

        # Initialize image list and index
        self.image_files = []
        self.current_image_index = -1

    def load_mask_images(self, folder_path):
        self.image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('masked_img_0.jpg')]
        self.image_files.sort()
        self.current_image_index = 0 if self.image_files else -1
        self.show_current_image()

    def load_images(self, folder_path):
        self.image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('png')]
        self.image_files.sort()
        self.current_image_index = 0 if self.image_files else -1
        self.show_current_image()

    def show_current_image(self):
        if self.image_files and 0 <= self.current_image_index < len(self.image_files):
            pixmap = QPixmap(self.image_files[self.current_image_index])
            scaled_pixmap = pixmap.scaled(self.image_display_area.size(), Qt.KeepAspectRatio)
            self.image_display_area.setPixmap(scaled_pixmap)
        else:
            self.image_display_area.clear()

    def show_previous_image(self):
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_current_image()

    def show_next_image(self):
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.show_current_image()

    def init3DDisplayArea(self, layout):
        show3dArea = QWidget()
        show3dLayout = QVBoxLayout()
        show3dArea.setLayout(show3dLayout)

        self.vtkWidget = QVTKRenderWindowInteractor(show3dArea)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1, 1, 1)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.createCone()
        self.interactor.Initialize()
        self.interactor.Start()

        show3dLayout.addWidget(self.vtkWidget)

        layout.addWidget(show3dArea, 1)

    def load_obj_model(self, file_path):
        file_lst = os.listdir(file_path)
        for file in file_lst:
            if file.endswith('obj'):
                obj_path = os.path.join(file_path, file)
                mesh = trimesh.load(obj_path)

                # Convert trimesh to VTK
                points = vtk.vtkPoints()
                cells = vtk.vtkCellArray()
                colors = vtk.vtkUnsignedCharArray()
                colors.SetNumberOfComponents(3)
                colors.SetName("Colors")

                for vertex in mesh.vertices:
                    points.InsertNextPoint(vertex[0], vertex[1], vertex[2])

                for face in mesh.faces:
                    triangle = vtk.vtkTriangle()
                    for i in range(3):
                        triangle.GetPointIds().SetId(i, face[i])
                    cells.InsertNextCell(triangle)

                for color in mesh.visual.vertex_colors:
                    colors.InsertNextTuple3(color[0], color[1], color[2])

                polydata = vtk.vtkPolyData()
                polydata.SetPoints(points)
                polydata.SetPolys(cells)
                polydata.GetPointData().SetScalars(colors)

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(polydata)

                actor = vtk.vtkActor()
                actor.SetMapper(mapper)

                self.renderer.RemoveAllViewProps()
                self.renderer.AddActor(actor)
                self.renderer.ResetCamera()
                self.vtkWidget.GetRenderWindow().Render()

    def initFilePathArea(self, layout):
        file_path_area = QWidget()
        file_path_layout = QVBoxLayout(file_path_area)

        # Create a horizontal layout to hold the file path label and browse button
        path_and_browse_layout = QHBoxLayout()

        # Create a label for the file path and add it to the horizontal layout
        self.file_path_label = QLabel("File Path:")
        path_and_browse_layout.addWidget(self.file_path_label)

        # Create a browse button and add it to the horizontal layout
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.uploadMesh)
        path_and_browse_layout.addWidget(self.browse_button)

        # Add the horizontal layout to the vertical layout
        file_path_layout.addLayout(path_and_browse_layout)

        # Create a QLineEdit for entering the file path and add it to the vertical layout
        self.file_path_entry = QLineEdit()
        self.file_path_entry.textChanged.connect(self.update_file_path_label)
        file_path_layout.addWidget(self.file_path_entry)

        # Add the file path area to the provided layout
        layout.addWidget(file_path_area)

    def initTextPromptArea(self, layout):
        # Create a QWidget to hold the text prompt area
        prompt_area = QWidget()
        prompt_layout = QVBoxLayout(prompt_area)
        # Create a horizontal layout to hold the label and submit button
        label_and_button_layout = QHBoxLayout()
        # Add a label for the text prompt
        self.prompt_label = QLabel("Text Prompt:")
        label_and_button_layout.addWidget(self.prompt_label)
        # Add a submit button
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.handleSubmit)
        label_and_button_layout.addWidget(self.submit_button)
        # Add the horizontal layout to the vertical layout
        prompt_layout.addLayout(label_and_button_layout)
        # Create and add the QLineEdit for text input
        self.prompt_entry = QLineEdit()
        prompt_layout.addWidget(self.prompt_entry)
        # Add the prompt area to the provided layout
        layout.addWidget(prompt_area)

    def initBottonArea(self, layout):
        Botton_area = QWidget()
        Botton_layout = QHBoxLayout(Botton_area)

        self.btnExportRes = QPushButton("Running")
        self.btnExportRes.clicked.connect(self.gen)
        Botton_layout.addWidget(self.btnExportRes)

        layout.addWidget(Botton_area)

    def initLogDisplayArea(self, layout):
        self.logArea = QTextEdit()
        self.logArea.setReadOnly(True)
        layout.addWidget(self.logArea)

        sys.stdout = EmittingStream(self.logArea)

    def uploadMesh(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "OBJ Files (*.obj);;All Files (*)", options=options)
        if file_path:
            self.file_path_entry.setText(file_path)

    def update_file_path_label(self):
        file_path = self.file_path_entry.text()

    def display_selected_image(self, current, previous):
        if current:
            image_path = current.data(Qt.UserRole)
            self.imageDisplayArea.show_image(image_path)

    def handleSubmit(self):
        self.input_prompt_text = self.prompt_entry.text()
        print(f"Text submitted: {self.input_prompt_text}")

    def gen(self):
        file_path = self.file_path_entry.text()
        prompt = self.prompt_entry.text()

        data_root = os.path.dirname(file_path)
        mesh_name = os.path.basename(file_path)

        self.logArea.clear()
        self.logArea.append("Rendering Multi-View Images\n")
        self.update()

        try:
            subprocess.run(
                # ["bash", "scripts/gen_render_img.sh", "-data_root", data_root, "-mesh_name", mesh_name, "-output_dir",
                #  OUTPUT_DIR, "-pose_file", POSE_FILE, "-step", "step_1"], check=True)
                ["python", "Inference.py", "-data_root", data_root, "-mesh_name", mesh_name, "-output_dir",
                 OUTPUT_DIR, "-pose_file", POSE_FILE, "-step", "step_1"], check=True)
        except subprocess.CalledProcessError as e:
            self.logArea.append(f"Error: {e}\n")
            self.update()
            return

        while not os.path.exists(STEP_1_DIR) or len(os.listdir(STEP_1_DIR)) != 8*2 + 1:
            time.sleep(10)

        self.load_images(STEP_1_DIR)

        self.logArea.append("Obtaining 2D Segmentation Mask\n")
        self.update()

        try:
            subprocess.run(
                # ["bash", "scripts/gen_mask.sh", "--image_path", STEP_1_DIR, "--prompt", prompt, "--vis_save_path",
                #  STEP_2_DIR], check=True)
                ["python", "2D_seg.py", "--image_path", STEP_1_DIR, "--prompt", prompt, "--vis_save_path",
                 STEP_2_DIR], check = True)
        except subprocess.CalledProcessError as e:
            self.logArea.append(f"Error: {e}\n")
            self.update()
            return

        while not os.path.exists(STEP_2_DIR) or len(os.listdir(STEP_2_DIR)) == 0:
            time.sleep(10)

        self.load_mask_images(STEP_2_DIR)

        self.logArea.append("Generating 3D Segmentation Mesh\n")
        self.update()

        try:
            subprocess.run(
                # ["bash", "scripts/gen_seg_mesh.sh", "-data_root", data_root, "-mesh_name", mesh_name, "-output_dir",
                #  STEP_3_DIR, "-step2_data", STEP_2_DIR, "-step", "step_3"], check=True)
                ["python", "scripts/Inference.py", "-data_root", data_root, "-mesh_name", mesh_name, "-output_dir",
                 STEP_3_DIR, "-step2_data", STEP_2_DIR, "-step", "step_3"], check=True)
        except subprocess.CalledProcessError as e:
            self.logArea.append(f"Error: {e}\n")
            self.update()
            return

        while not os.path.exists(STEP_3_DIR) or not any(file.endswith('.obj') for file in os.listdir(STEP_3_DIR)):
            time.sleep(10)

        self.logArea.append("Completed\n")
        self.load_obj_model(STEP_3_DIR)

    def gen_test(self):
        self.load_images(STEP_2_DIR)
        self.load_obj_model(STEP_3_DIR)

    def update(self):
        QApplication.processEvents()

    def view_check(self, view_type):
        if view_type == "left":
            self.renderer.GetActiveCamera().Azimuth(90)
        elif view_type == "front":
            self.renderer.GetActiveCamera().Azimuth(0)
        elif view_type == "down":
            self.renderer.GetActiveCamera().Elevation(-90)
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    def createCone(self):
        cone = vtk.vtkConeSource()
        cone.SetHeight(3.0)
        cone.SetRadius(1.0)
        cone.SetResolution(10)

        coneMapper = vtk.vtkPolyDataMapper()
        coneMapper.SetInputConnection(cone.GetOutputPort())

        coneActor = vtk.vtkActor()
        coneActor.SetMapper(coneMapper)

        self.renderer.AddActor(coneActor)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
