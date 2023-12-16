import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from scipy.ndimage import gaussian_filter
import caiman as cm

class VideoPlayer(QtWidgets.QWidget):
    def __init__(self, data):
        super(VideoPlayer, self).__init__()

        # Data and frame control
        self.data = data
        self.current_frame = 0
        self.playback_timer = QtCore.QTimer(self)
        self.playback_timer.timeout.connect(self.next_frame)

        # ImageView setup
        self.image_view = pg.ImageView(view=pg.PlotItem())
        self.image_view.ui.histogram.gradient.setColorMap(pg_cmap)

        # Set initial levels
        self.levels = (np.min(self.data), np.max(self.data))
        self.image_view.setImage(self.data[0], levels=self.levels)

        # Play/pause button
        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)

        # Frame slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(data) - 1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_display)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.image_view)
        layout.addWidget(self.play_button)
        layout.addWidget(self.slider)
        self.setLayout(layout)

    def next_frame(self):
        if self.current_frame + 1 < len(self.data):
            self.current_frame += 1
            self.update_display()
        else:
            self.playback_timer.stop()

    def toggle_playback(self):
        if self.playback_timer.isActive():
            self.playback_timer.stop()
            self.play_button.setText("Play")
        else:
            self.playback_timer.start(1000 // 30)  # Assuming 30 FPS, adjust accordingly
            self.play_button.setText("Pause")

    def update_display(self):
        # Read the current levels from the ImageView
        self.levels = self.image_view.getHistogramWidget().getLevels()

        self.current_frame = self.slider.value()
        self.image_view.setImage(self.data[self.current_frame], levels=self.levels)
        self.slider.setValue(self.current_frame)


# Set up colormap
colors = [(0, 0, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 255, 255)]
positions = [0, 0.25, 0.5, 0.75, 1]
pg_cmap = pg.ColorMap(positions, colors)

# Sample data generation
frames = 100
# data = [gaussian_filter(np.random.normal(size=(512, 512)), 4) for _ in range(frames)]
mmap_path = r'C:\Users\ba81\caiman_data\example_movies\1_memmap_d1_340_d2_365_d3_1_order_C_frames_1000.mmap'
Yr, dims, T = cm.load_memmap(mmap_path)
images = Yr.T.reshape((T,) + dims, order='F')
data = np.array(images)

app = QtWidgets.QApplication([])
window = VideoPlayer(data)
window.show()
app.exec_()
