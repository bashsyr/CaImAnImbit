#%% Imports
import seaborn
import matplotlib.pyplot as plt
    import numpy as np

    #different matplotlib backend for better plots
    import matplotlib
    matplotlib.rcParams['backend'] = 'TkAgg'
    ##
    import caiman as cm
    #from caiman.source_extraction import cnmf
    from caiman.utils.utils import download_demo
    from caiman.utils.visualization import inspect_correlation_pnr
    #from caiman.motion_correction import MotionCorrect
    #from caiman.source_extraction.cnmf import params as params

    # Other imports
    import glob
    import time
    import natsort
    import pickle as pkl


#%% Test for the automatic PNR CORR calculation

    results = {}
    times = []
    path_list = []
    cn_filters = []
    pnr_list = []
    frame_increments = []


    results_file = r'O:\archive\projects\2023_students\Result_files\Test_summary_image\m1020.pkl'


     paths  = [#r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230705_m1310_som_1410\miniscope_video\*',
    #           r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230704_m1310_som_1939\miniscope_video\*',
    #           r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230627_m1310_som_1325\miniscope_video\*',
    #           r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230620_m1310_som_1711\miniscope_video\*',
              # r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230619_m1310_som_0947\miniscope_video\*',
              # r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230616_m1310_som_1554\miniscope_video\*',
              # r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230707_m1309_som_1152\miniscope_video\*',
              # r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230612_m1309_som_1302\miniscope_video\*',
              # r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230802_m1309_som_1746\miniscope_video\*',
              # r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230705_m1309_som_1453\miniscope_video\*',
              # r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230707_m1308_som_1520\miniscope_video\*',
              # # r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230615_m1308_som_0936\miniscope_video\*',
              # # # r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230616_m1310_som_1150\miniscope_video\*',
              # r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230705_m1308_som_1534\miniscope_video\*',
              # r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230608_m1308_som_1636\miniscope_video\*',
              # r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230704_m1308_som_1813\miniscope_video\*',
              # r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230608_m1310_som_1357\miniscope_video\*',
              # r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230621_m1310_som_0933\miniscope_video\*',

              # r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230809_m1310_som_1243\miniscope_video\*',
              # r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230801_m1310_som_1056\miniscope_video\*',
              # r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230728_m1310_som_1747\miniscope_video\*'
                r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230420_m1020_wt_1658\miniscope_video\*',
                r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230425_m1020_wt_1524\miniscope_video\*',
                r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230321_m1020_wt_1622\miniscope_video\*',
                r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230329_m1020_wt_1500\miniscope_video\*',
              ]


    for path in paths:
        files_names = glob.glob(path + r'F_frames*1000*.mmap')
        files_names = natsort.natsorted(files_names)
        for r in range(0,5):
            file_increment = 5
            # images = []
            st = time.time()
            # memmap_list = []  # a list of the individual videos as memmaps
            # for index, name in enumerate(files_names):
            #     Yr, dims, T = cm.load_memmap(name, mode='r')
            #     memmap_list.append(Yr.T)
            gSig = (5, 5)
            images = cm.load_movie_chain(files_names[r::file_increment])
            try:
                # images = np.concatenate(([item for item in memmap_list[r::file_increment]]), axis=0)
                # images = images.reshape(len(images), dims[0], dims[1], order='F')
                cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=gSig[0], swap_dim=False)
            except:
                passse
            et = time.time()
            elapsed_time = et - st
            print('Execution time:', elapsed_time, 'seconds')
            file_increment = str(file_increment)+ '_' + str(r)

            frame_increments.append(file_increment)
            times.append(elapsed_time)
            cn_filters.append(cn_filter)
            pnr_list.append(pnr)
            path_list.append(path)
            # set for the next round
            del images
            # del memmap_list

    # Save results in file
    results['frame_increments'] = frame_increments
    results['times'] = times
    results['paths'] = path_list
    results['cn_filters'] = cn_filters
    results['pnr_list'] = pnr_list
    pkl.dump(results, open(results_file, "wb"))



    #
    results = pkl.load(open(results_file, "rb"))
    times = results['times']
    path_list =  results['paths']
    cn_filters =  results['cn_filters']
    pnr_list =  results['pnr_list']
    frame_increments = results['frame_increments']


























#%% test code to make the DB calulate Summary image on a seet of F_ordered memmaps instead of the big C_ordered one
    st = time.time()
    # path = r'D:\CaImAn_Data\data\1_preprocessed\20230405_m1018_wt_1110\*'          # path to the result files
    path = r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230405_m1018_wt_1110\miniscope_video\*'         # sever path
    frame_increment = 5         # downsamppling parameter
    files_names = glob.glob(path + r'*F_frames*1000.mmap')
    files_names = natsort.natsorted(files_names)
    memmap_list = []    # a list of the individual videos as memmaps
    for index,name in enumerate(files_names):
        Yr, dims, T = cm.load_memmap(name, mode='r')
        memmap_list.append(Yr.T)
    gSig = (5, 5)
    images = np.concatenate(([item for item in memmap_list[::frame_increment]]),axis = 0)
    images = images.reshape(len(images),dims[0],dims[1],order='F')
    cn_filter,pnr = cm.summary_images.correlation_pnr(images[::2], gSig=gSig[0], swap_dim=False)
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    # cm.utils.visualization.inspect_correlation_pnr(cn_filter, pnr)
import cv2

cv2.VideoCapture()

#%% TEST the test code for Summary image
    results = {}
    times = []
    frame_increments = []
    cn_filters = []
    pnr_list = []

    # path = r'D:\CaImAn_Data\data\1_preprocessed\20230405_m1018_wt_1110\*'          # path to the result files
    path = r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230405_m1018_wt_1110\miniscope_video\*'  # sever path
    files_names = glob.glob(path + r'*F_frames*1000.mmap')
    files_names = natsort.natsorted(files_names)


    for frame_increment in range(10,0,-1):
        images = []
        st = time.time()
        memmap_list = []  # a list of the individual videos as memmaps
        for index, name in enumerate(files_names):
            Yr, dims, T = cm.load_memmap(name, mode='r')
            memmap_list.append(Yr.T)
        gSig = (5, 5)
        try:
            images = np.concatenate(([item for item in memmap_list[::frame_increment]]), axis=0)
            images = images.reshape(len(images), dims[0], dims[1], order='F')
            cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=gSig[0], swap_dim=False)
        except:
            pass
        et = time.time()
        elapsed_time = et - st
        print('Execution time:', elapsed_time, 'seconds')

        # #collect results
        # results = pkl.load(open("ResultsDB_F_Frame_Files.p", "rb"))
        # times = results['times']
        # frame_increments =  results['frame_increments']
        # cn_filters =  results['cn_filters']
        # pnr_list =  results['pnr_list']

        times.append(elapsed_time)
        cn_filters.append(cn_filter)
        pnr_list.append(pnr)
        frame_increments.append(frame_increment)

        # set for the next round
        del images
        del memmap_list
    # do a safety save before testing the big file
    results['times'] = times
    results['frame_increments'] = frame_increments
    results['cn_filters'] = cn_filters
    results['pnr_list'] = pnr_list
    pkl.dump( results, open( "ResultsDB_F_Frame_Files_1.p", "wb" ) )

    # Test the big original file

    video_for_calc = cm.load(r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230405_m1018_wt_1110\miniscope_video\memmap__d1_341_d2_398_d3_1_order_C_frames_37564.mmap')
    gsig = 5
    for frame_increment in range(1,0,-1):
        st = time.time()

        try:
            cn_filter, pnr = cm.summary_images.correlation_pnr(video_for_calc[::frame_increment], gSig=gsig, swap_dim=False)
        except:
            cn_filter = [0]
            pnr = [0]

        et = time.time()
        elapsed_time = et - st
        times.append(elapsed_time)
        cn_filters.append(cn_filter)
        pnr_list.append(pnr)
        frame_increments.append(100000+frame_increment)

    results['times'] = times
    results['frame_increments'] = frame_increments
    results['cn_filters'] = cn_filters
    results['pnr_list'] = pnr_list
    pkl.dump( results, open( "ResultsDB_F_Frame_Files_2.p", "wb" ) )

    # os.system("shutdown /s /t 1")


    # pkl.load( open ("ResultsDB_F_Frame_Files.p", "rb")

    # cm.utils.visualization.inspect_correlation_pnr(cn_filter, pnr)
    # cm.utils.visualization.inspect_correlation_pnr(cn_filters[i], pnr_list[i])
    # seaborn.kdeplot(cn_filter.flatten())



def calculate_minCORR(cn_filters: list):
    # do trough detection to calculate min_corr
    Troughs = []
    for i in range(len(cn_filters)):

        data = cn_filters[i].flatten()
        kde = gaussian_kde(data)

        # Define the range for evaluating KDE
        x = np.linspace(.3, max(data), 1000)  # .2 to prevent it from detecting peaks at the beginning
        y = kde.evaluate(x)

        # Find peaks
        peaks, _ = find_peaks(y, distance=20)  # 'distance' may need adjustment based on your data

        # Find the trough between the main peaks (assuming the two main peaks are the first two found)
        if len(peaks) > 1:
            trough = np.argmin(y[peaks[0]:peaks[1]]) + peaks[0]
        else:
            trough = None

        # Visualize
        plt.figure(i)           # to print plots individually
        plt.plot(x, y, label='KDE')
        plt.plot(x[peaks], y[peaks], "x", label='Peaks')
        if trough is not None:
            plt.plot(x[trough], y[trough], "o", label='Trough')
        plt.legend()
        plt.show()

        if trough is not None:
            # print(f'Trough value: {x[trough]}')
            Troughs.append(x[trough])

        # else:
        #     print("Could not identify a clear trough between peaks.")

    Troughs = [round(i, 2) for i in Troughs]
    print(Troughs)
    min_corr = np.mean(Troughs)
    return min_corr


def calculate_minCORR(cn_filters: list):
    # do trough detection to calculate min_corr
    Troughs = []
    for i in range(len(cn_filters)):

        data = cn_filters[i].flatten()
        kde = gaussian_kde(data)

        # Define the range for evaluating KDE
        x = np.linspace(min(data), max(data), 1000)  # .2 to prevent it from detecting peaks at the beginning
        y = kde.evaluate(x)

        # Find peaks
        peaks, _ = find_peaks(y, distance=20)  # 'distance' may need adjustment based on your data

        # Find the trough between the main peaks (assuming the two main peaks are the first two found)
        if len(peaks) > 1:
            highest_peak_idx = np.argmax(y[peaks])
            trough = np.argmin(y[peaks[highest_peak_idx]:peaks[-1]]) + peaks[highest_peak_idx]
            if y[trough] + .2 > y[peaks[-1]] :
                trough = None


        else:
            trough = None

        # Visualize
        plt.figure(i)           # to print plots individually
        plt.plot(x, y, label='KDE')
        plt.plot(x[peaks], y[peaks], "x", label='Peaks')
        if trough is not None:
            plt.plot(x[trough], y[trough], "o", label='Trough')
        plt.legend()
        plt.show()

        if trough is not None:
            # print(f'Trough value: {x[trough]}')
            Troughs.append(x[trough])

        # else:
        #     print("Could not identify a clear trough between peaks.")

    Troughs = [round(i, 2) for i in Troughs]
    print(Troughs)
    if Troughs:
        min_corr = np.mean(Troughs)
    else:
        print('Auto min_corr selection failed, set to default value [min_corr = .85] and flag for human inspection')
        min_corr = .85
    return min_corr






from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QSlider, QLabel, QPushButton
from PyQt5.QtCore import Qt, QTimer
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import caiman as cm
class DualSlider(QWidget):
    def __init__(self, min_val=0, max_val=255):
        super().__init__()
        self.layout = QHBoxLayout()
        self.min_slider = QSlider(Qt.Horizontal)
        self.max_slider = QSlider(Qt.Horizontal)
        self.min_slider.setRange(min_val, max_val)
        self.max_slider.setRange(min_val, max_val)
        self.min_slider.setValue(min_val)
        self.max_slider.setValue(max_val)
        self.layout.addWidget(self.min_slider)
        self.layout.addWidget(self.max_slider)
        self.setLayout(self.layout)

class App(QMainWindow):
    def __init__(self, movie_data, filter_function):
        super().__init__()
        self.movie_data = movie_data
        self.filter_function = filter_function
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        self.canvas = FigureCanvas(plt.figure(figsize=(10, 5)))
        self.ax = self.canvas.figure.subplots(1, 2)
        layout.addWidget(self.canvas)
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.next_frame)
        self.play_timer.start(20)  # for roughly 50 FPS
        play_button = QPushButton("Play", self)
        pause_button = QPushButton("Pause", self)
        play_button.clicked.connect(self.play)
        pause_button.clicked.connect(self.pause)
        button_layout = QHBoxLayout()
        button_layout.addWidget(play_button)
        button_layout.addWidget(pause_button)
        layout.addLayout(button_layout)
        self.frame_slider = QSlider(Qt.Horizontal, self)
        self.frame_slider.setRange(0, len(self.movie_data) - 1)
        self.frame_slider_label = QLabel("Frame: 0", self)
        layout.addWidget(self.frame_slider_label)
        layout.addWidget(self.frame_slider)
        self.gSig_slider = QSlider(Qt.Horizontal, self)
        self.gSig_slider.setRange(1, 33)
        self.gSig_slider.setSingleStep(2)
        self.gSig_slider_label = QLabel("gSig_filt: 1", self)
        layout.addWidget(self.gSig_slider_label)
        layout.addWidget(self.gSig_slider)
        self.orig_brightness = DualSlider(0, 255)
        self.filtered_brightness = DualSlider(-2, 3)
        layout.addWidget(QLabel("Original Brightness", self))
        layout.addWidget(self.orig_brightness)
        layout.addWidget(QLabel("Filtered Brightness", self))
        layout.addWidget(self.filtered_brightness)
        self.frame_slider.valueChanged.connect(self.update_display)
        self.gSig_slider.valueChanged.connect(self.update_display)
        self.orig_brightness.min_slider.valueChanged.connect(self.update_display)
        self.orig_brightness.max_slider.valueChanged.connect(self.update_display)
        self.filtered_brightness.min_slider.valueChanged.connect(self.update_display)
        self.filtered_brightness.max_slider.valueChanged.connect(self.update_display)
        self.update_display()
        self.setWindowTitle('Movie Player')
        self.show()

    def play(self):
        self.play_timer.start()

    def pause(self):
        self.play_timer.stop()

    def next_frame(self):
        current_frame = (self.frame_slider.value() + 1) % len(self.movie_data)
        self.frame_slider.setValue(current_frame)
        self.update_display()

    def update_display(self):
        current_frame = self.frame_slider.value()
        self.frame_slider_label.setText(f"Frame: {current_frame}")
        gSig_filt = (self.gSig_slider.value(), self.gSig_slider.value())
        self.gSig_slider_label.setText(f"gSig_filt: {gSig_filt[0]}")
        filtered_frame = self.filter_function(self.movie_data[current_frame], gSig_filt)
        orig_vmin = min(self.orig_brightness.min_slider.value(), self.orig_brightness.max_slider.value())
        orig_vmax = max(self.orig_brightness.min_slider.value(), self.orig_brightness.max_slider.value())
        filt_vmin = min(self.filtered_brightness.min_slider.value(), self.filtered_brightness.max_slider.value())
        filt_vmax = max(self.filtered_brightness.min_slider.value(), self.filtered_brightness.max_slider.value())
        self.ax[0].imshow(self.movie_data[current_frame], cmap='gnuplot2', vmin=orig_vmin, vmax=orig_vmax)
        self.ax[1].imshow(filtered_frame, cmap='gnuplot2', vmin=filt_vmin, vmax=filt_vmax)
        self.canvas.draw()


if __name__ == '__main__':
    # load the movie file
    mmap_path = r'C:\Users\ba81\caiman_data\example_movies\1_memmap_d1_340_d2_365_d3_1_order_C_frames_1000.mmap'
    Yr, dims, T = cm.load_memmap(mmap_path)
    images = Yr.T.reshape((T,) + dims, order='F')
    movie_data = images
    def dummy_filter(frame, gSig_filt):
        return high_pass_filter_space(frame,gSig_filt)  # Modify as per your actual filter function

    app = QApplication(sys.argv)

    def dummy_filter(frame, gSig_filt):  # dummy filter function
        return frame * gSig_filt[0]
    ex = App(movie_data, dummy_filter)
    sys.exit(app.exec_())



#%% code to get the mean shifts graph them, and do pearson correlation
import pickle
import numpy as np
import matplotlib
matplotlib.rcParams['backend'] = 'Qt5Agg'  # works for win 11
import matplotlib.pyplot as plt

with open(r'O:\archive\projects\2023_students\Result_files\shifts\1_1.pickle', 'rb') as handle:
    shifts = pickle.load(handle)


keys, values = zip(*shifts.items())

keys = list(keys)
keys1 = keys
x = [[values[j][i][0] for i in range(1000)]for j in range (len(values))]
y = [[values[j][i][1] for i in range(1000)]for j in range (len(values))]

xy_shifts = [[np.abs(values[j][i][0]) + np.abs(values[j][i][1]) for i in range(1000)]for j in range (len(values))]


#%% plot the mean shift against gSig_filt
means = [np.mean(np.abs(values[i])) for i in range(len(values))]
plt.plot(list(keys),np.mean(xy_shifts,axis=1))


#%% plot mean shift
import numpy as np
import matplotlib.pyplot as plt

# Assuming your data is in the format: data = [[shifts_param1], [shifts_param2], ..., [shifts_param44]]
# data = [... your data ...]
# And keys = [param1, param2, ..., param44]
data = xy_shifts
keys = keys1
# Convert data to a NumPy array for easier manipulation
# data = np.array(data[::2])
# keys = np.array(keys[::2])

# Calculate differences between consecutive shifts for each parameter
diffs = np.diff(data, axis=0)

# Calculate the mean of differences considering the one before and the one after
mean_diffs = []
for i in range(1, len(diffs)):
    mean_diff = np.mean(np.abs(diffs[max(0, i-1):min(i+1, len(diffs))]), axis=0)
    mean_diffs.append(np.mean(mean_diff))

# Plotting
plt.figure(figsize=(15, 7))
for i, val in enumerate(mean_diffs):
    plt.plot([keys[i+1], keys[i+1]], [0, val], 'b-', linewidth=0.8)  # Vertical lines
    plt.plot(keys[i+1], val, 'ro')  # Red dot at the top of each line

plt.xlabel('gSig_filt Value',fontsize=17)
plt.ylabel('Mean Absolute Difference of Shifts',fontsize=17)
plt.title('Change in Shifts Across gSig_filt Values',fontsize=17)
plt.xticks(fontsize=13)  # Adjust the fontsize as needed
plt.yticks(fontsize=15)  # Adjust the fontsize as needed
plt.grid(True)
plt.savefig('mean_absulote_difference.pdf')
plt.show()





#%% plot the mean shift against gSig_filt
means = [np.mean(np.abs(values[i])) for i in range(len(values))]
plt.plot(list(keys),means)
plt.xlabel('gSig_filt Value',fontsize=16)
plt.ylabel('Mean Total Shifts ',fontsize=16)
plt.title('Change in Shifts Across gSig_filt Values',fontsize=16)
# Increase the tick label font size on both axes
plt.xticks(fontsize=13)  # Adjust the fontsize as needed
plt.yticks(fontsize=15)  # Adjust the fontsize as needed
plt.savefig('found_movement.pdf')


sorted_cnm[101].estimates.plot_contours()

# Calculate the Pearson correlation
correlation_coefficient = np.corrcoef(x[3], x[6])[0, 1]

print(f"Pearson correlation coefficient: {correlation_coefficient:.3f}")

