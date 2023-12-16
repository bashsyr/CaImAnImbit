import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QPushButton, QGridLayout, QApplication
from PyQt5.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from caiman.motion_correction import high_pass_filter_space
import caiman as cm
class App(QWidget):
    def __init__(self, movie_data, filter_function):
        super().__init__()
        self.movie_data = movie_data
        self.filter_function = filter_function
        self.current_frame = 0
        self.init_ui()

        # Timer for movie playback
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

    def init_ui(self):
        self.layout = QVBoxLayout()

        # Display for movie frames
        self.fig, self.ax = plt.subplots(1, 2)
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        # Playback buttons
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.start_playback)
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_playback)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.play_button)
        buttons_layout.addWidget(self.pause_button)
        self.layout.addLayout(buttons_layout)

        # Frame slider with label
        self.frame_label = QLabel("Frame: 0")
        self.frame_slider = QSlider()
        self.frame_slider.setRange(0, len(self.movie_data) - 1)
        self.frame_slider.valueChanged.connect(self.show_frame)

        frame_layout = QVBoxLayout()
        frame_layout.addWidget(self.frame_label)
        frame_layout.addWidget(self.frame_slider)
        self.layout.addLayout(frame_layout)

        # gSig_filt slider with label
        self.gSig_label = QLabel("gSig_filt: 1")
        self.gSig_slider = QSlider()
        self.gSig_slider.setRange(1, 33)
        self.gSig_slider.setValue(1)
        self.gSig_slider.valueChanged.connect(self.update_display)

        gSig_layout = QVBoxLayout()
        gSig_layout.addWidget(self.gSig_label)
        gSig_layout.addWidget(self.gSig_slider)
        self.layout.addLayout(gSig_layout)

        # Brightness sliders with labels based on data range
        data_min, data_max = np.min(self.movie_data), np.max(self.movie_data)

        # Sliders for Original Movie Brightness
        data_min, data_max = np.min(self.movie_data), np.max(self.movie_data)

        # Create a grid layout for sliders
        sliders_grid = QGridLayout()

        # Original Brightness Sliders
        orig_label = QLabel("Original Brightness")
        sliders_grid.addWidget(orig_label, 0, 0, 1, 2)

        self.orig_vmin_slider = QSlider()
        self.orig_vmax_slider = QSlider()
        self.orig_vmin_slider.setOrientation(Qt.Horizontal)
        self.orig_vmax_slider.setOrientation(Qt.Horizontal)
        sliders_grid.addWidget(self.orig_vmin_slider, 1, 0)
        sliders_grid.addWidget(self.orig_vmax_slider, 1, 1)

        # Filtered Brightness Sliders
        filt_label = QLabel("Filtered Brightness")
        sliders_grid.addWidget(filt_label, 2, 0, 1, 2)

        self.filt_vmin_slider = QSlider()
        self.filt_vmax_slider = QSlider()
        self.filt_vmin_slider.setOrientation(Qt.Horizontal)
        self.filt_vmax_slider.setOrientation(Qt.Horizontal)
        sliders_grid.addWidget(self.filt_vmin_slider, 3, 0)
        sliders_grid.addWidget(self.filt_vmax_slider, 3, 1)

        self.layout.addLayout(sliders_grid)
        self.setLayout(self.layout)
        self.update_display()

    def start_playback(self):
        self.timer.start(20)  # Update every 100ms

    def pause_playback(self):
        self.timer.stop()

    def next_frame(self):
        if self.current_frame < len(self.movie_data) - 1:
            self.current_frame += 1
            self.frame_slider.setValue(self.current_frame)

    def show_frame(self, frame_idx):
        self.current_frame = frame_idx
        self.frame_label.setText(f"Frame: {frame_idx}")
        self.update_display()


    def update_display(self):
        gSig_filt = self.gSig_slider.value()
        self.gSig_label.setText(f"gSig_filt: {gSig_filt}")

        # vmin = self.vmin_slider.value()
        # self.vmin_label.setText(f"Brightness Min: {vmin}")
        #
        # vmax = self.vmax_slider.value()
        # self.vmax_label.setText(f"Brightness Max: {vmax}")


        orig_vmin = self.orig_vmin_slider.value()
        orig_vmax = self.orig_vmax_slider.value()

        filt_vmin = self.filt_vmin_slider.value() / 100.0  # Converting back to float
        filt_vmax = self.filt_vmax_slider.value() / 100.0

        frame = self.movie_data[self.current_frame]
        filtered_frame = self.filter_function(frame, gSig_filt)

        # Display the images and get their references
        # im1 = self.ax[0].imshow(frame, cmap="gnuplot2")
        # im2 = self.ax[1].imshow(filtered_frame, cmap="gnuplot2")
        im1 = self.ax[0].imshow(frame, vmin=orig_vmin, vmax=orig_vmax, cmap="gnuplot2")
        im2 = self.ax[1].imshow(filtered_frame, vmin=filt_vmin, vmax=filt_vmax, cmap="gnuplot2")

        # # Set color limits for each image
        # im1.set_clim(vmin, vmax)
        # im2.set_clim(vmin, vmax)

        self.canvas.draw()


if __name__ == "__main__":
    # load the movie file
    mmap_path = r'C:\Users\ba81\caiman_data\example_movies\1_memmap_d1_340_d2_365_d3_1_order_C_frames_1000.mmap'
    Yr, dims, T = cm.load_memmap(mmap_path)
    images = Yr.T.reshape((T,) + dims, order='F')
    input_movie = images
    input_movie = np.array(input_movie)

    movie_data = images
    def dummy_filter(frame, gSig_filt):
        return high_pass_filter_space(frame,(gSig_filt,gSig_filt))  # Modify as per your actual filter function

    app = QApplication(sys.argv)
    ex = App(movie_data, dummy_filter)
    ex.show()
    sys.exit(app.exec_())