import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore, QtWidgets
from pyqtgraph import ColorMap
from scipy.ndimage.filters import gaussian_filter

import caiman as cm
from caiman.motion_correction import high_pass_filter_space
class VideoPlayer(QtWidgets.QWidget):
    def __init__(self, data):
        super(VideoPlayer, self).__init__()
        self.data = data
        self.current_frame = 0
        self.playing = False
        self.levels = (-1, 3)
        # Layout
        layout = QtWidgets.QVBoxLayout()

        # Image view
        self.image_view = pg.ImageView()
        layout.addWidget(self.image_view)

        # Slider for frame navigation
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, len(self.data) - 1)
        self.slider.valueChanged.connect(self.update_display)
        layout.addWidget(self.slider)

        # gSig_filt slider
        self.gSig_filt_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gSig_filt_slider.setRange(1, 33)
        self.gSig_filt_slider.setValue(5)
        self.gSig_filt_slider.setSingleStep(2)
        self.gSig_filt_slider.valueChanged.connect(self.apply_filter)
        self.gSig_filt_slider.valueChanged.connect(self.ensure_odd_gSig)

        layout.addWidget(self.gSig_filt_slider)

        # Play/pause button
        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        layout.addWidget(self.play_button)
        self.setLayout(layout)


        # Create and set colormap (blue -> red -> yellow)
        colors = [(0, 0, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 255, 255)]
        positions = [0, 0.25, 0.5, 0.75, 1]
        colormap = ColorMap(positions, colors)
        self.image_view.setColorMap(colormap)


        # Display the frame number and gSig_filt value
        self.info_label = QtWidgets.QLabel()
        layout.addWidget(self.info_label)

        # button to save the gSig_filt value
        self.save_button = QtWidgets.QPushButton("Save gSig_filt")
        layout.addWidget(self.save_button)
        self.save_button.clicked.connect(self.save_gSig_filt)

        # QTimer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.advance_frame)

        self.show_frame(self.current_frame)

    def apply_filter(self):
        self.show_frame(self.current_frame)
        self.update_info_label()
    def show_frame(self, frame_num):
        gSig_filt = (self.gSig_filt_slider.value(), self.gSig_filt_slider.value())
        frame = high_pass_filter_space(self.data[frame_num], gSig_filt)
        # self.image_view.setImage(frame.T)
        self.levels = self.image_view.getHistogramWidget().getLevels()
        self.image_view.setImage(frame.T, levels=self.levels)
        # self.levels = self.image_view.getHistogramWidget().getLevels()
        self.slider.setValue(frame_num)
        self.update_info_label()
    def advance_frame(self):
        if self.current_frame < len(self.data) - 1:
            self.current_frame += 1
        else:
            self.timer.stop()
            self.playing = False
            self.play_button.setText("Play")
        self.show_frame(self.current_frame)

    def toggle_playback(self):
        if self.playing:
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.timer.start(int(1000 / 70))
            self.play_button.setText("Pause")
        self.playing = not self.playing

    def update_display(self):
        self.current_frame = self.slider.value()
        self.show_frame(self.current_frame)
    def update_info_label(self):
        info_text = f"Frame: {self.current_frame + 1}/{len(self.data)} | gSig_filt: {self.gSig_filt_slider.value()}"
        self.info_label.setText(info_text)

    def ensure_odd_gSig(self):
        value = self.gSig_filt_slider.value()
        if value % 2 == 0:
            # Simply increment the value to make it odd
            value += 1
            self.gSig_filt_slider.setValue(value)

    def save_gSig_filt(self):
        self.gSig_filt = self.gSig_filt_slider.value()
        print(f"gSig_filt saved as: {self.gSig_filt}")

    def get_gSig_value(self):
        return self.gSig_filt
if __name__ == '__main__':
    # movie_data = np.random.random((100, 512, 512))
    # load the movie file
    mmap_path = r'C:\Users\ba81\caiman_data\example_movies\1_memmap_d1_340_d2_365_d3_1_order_C_frames_1000.mmap'
    Yr, dims, T = cm.load_memmap(mmap_path)
    images = Yr.T.reshape((T,) + dims, order='F')



    app = QtWidgets.QApplication([])
    window = VideoPlayer(images)
    window.show()
    app.exec_()
