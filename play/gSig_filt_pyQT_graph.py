import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider
from PyQt5.QtCore import Qt
import caiman as cm
class VideoPlayer(QMainWindow):
    def __init__(self, movie_data):
        super().__init__()

        # Set up the main window layout and central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout()
        self.central_widget.setLayout(layout)

        # Video data
        self.movie_data = movie_data
        self.current_frame = 0

        # Image view for displaying video
        self.image_view = pg.ImageView(view=pg.PlotItem())
        layout.addWidget(self.image_view)

        # Slider for navigating through video frames
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(len(movie_data) - 1)
        self.frame_slider.valueChanged.connect(self.update_frame)
        layout.addWidget(self.frame_slider)

        # gSig_filt Slider
        self.gsig_slider = QSlider(Qt.Horizontal)
        self.gsig_slider.setMinimum(1)
        self.gsig_slider.setMaximum(33)
        self.gsig_slider.setSingleStep(2)
        self.gsig_slider.valueChanged.connect(self.apply_filter)
        layout.addWidget(self.gsig_slider)

        self.show()

    def update_frame(self, frame_index):
        self.current_frame = frame_index
        self.image_view.setImage(self.movie_data[frame_index])

    def apply_filter(self, gsig_value):
        # Here, apply the filter function to your video data
        # You may want to modify this as per your filter's requirements
        # gSig_filt = (gsig_value, gsig_value)
        # filtered_frame = high_pass_filter_space(self.movie_data[self.current_frame], gSig_filt)
        # You would then display the filtered frame using:
        # self.image_view.setImage(filtered_frame)
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Dummy movie data for demonstration
    # movie_data = np.random.random((100, 256, 256))  # 100 frames of 256x256

    # load the movie file
    mmap_path = r'C:\Users\ba81\caiman_data\example_movies\1_memmap_d1_340_d2_365_d3_1_order_C_frames_1000.mmap'
    Yr, dims, T = cm.load_memmap(mmap_path)
    images = Yr.T.reshape((T,) + dims, order='F')
    movie_data = np.array(images)

    window = VideoPlayer(movie_data)
    sys.exit(app.exec_())
