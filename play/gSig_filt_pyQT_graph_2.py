import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore, QtWidgets

from scipy.ndimage.filters import gaussian_filter

import caiman as cm
from caiman.motion_correction import high_pass_filter_space
class VideoPlayer(QtWidgets.QWidget):
    def __init__(self, data):
        super(VideoPlayer, self).__init__()
        self.data = data
        self.current_frame = 0
        self.playing = False

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
        layout.addWidget(self.gSig_filt_slider)

        # Play/pause button
        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        layout.addWidget(self.play_button)

        self.setLayout(layout)

        # QTimer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.advance_frame)

        self.show_frame(self.current_frame)

    def apply_filter(self):
        self.show_frame(self.current_frame)

    def show_frame(self, frame_num):
        gSig_filt = (self.gSig_filt_slider.value(), self.gSig_filt_slider.value())
        frame = high_pass_filter_space(self.data[frame_num], gSig_filt)
        self.image_view.setImage(frame.T)
        self.slider.setValue(frame_num)

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
            self.timer.start(int(1000 / 30))
            self.play_button.setText("Pause")
        self.playing = not self.playing

    def update_display(self):
        self.current_frame = self.slider.value()
        self.show_frame(self.current_frame)



if __name__ == '__main__':
    # movie_data = np.random.random((100, 512, 512))
    # load the movie file
    mmap_path = r'C:\Users\ba81\caiman_data\example_movies\1_memmap_d1_340_d2_365_d3_1_order_C_frames_1000.mmap'
    Yr, dims, T = cm.load_memmap(mmap_path)
    images = Yr.T.reshape((T,) + dims, order='F')
    movie_data = np.array(images)


    app = QtWidgets.QApplication([])
    window = VideoPlayer(movie_data)
    window.show()
    app.exec_()
