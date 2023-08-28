"""
Code to estimate the value of Session specific Caiman parameters (minPNR, minSNR)
Author: Bashar
"""

#%% Imports
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

#%% import Memmap files
    # task specific parameters
    path = r'D:\CaImAn_Data\data\1_preprocessed\20230405_m1018_wt_1110\*'# path to the result files
    # path = r'O:\archive\projects\2023_students\TestExperiment\data\1_preprocessed\20230405_m1018_wt_1110\miniscope_video\*'         # sever path

    result_files_names = glob.glob(path + r'*F_frames*1000.mmap')
    result_files_names = natsort.natsorted(result_files_names)
    images = []    # a list of the individual videos a memmaps
    for index,name in enumerate(result_files_names):
        Yr, dims, T = cm.load_memmap(name, mode='r+')
        images.append(Yr.T.reshape((T,) + dims, order='F'))
    gSig = (5, 5)
    cn_filter = []
    pnr = []
    images = images[::]                # to make the list shorter for faster testing
    st = time.time()
    for index,image in enumerate(images):
        res = cm.summary_images.correlation_pnr(image[::1], gSig=gSig[0], swap_dim=False)
        cn_filter.append (res[0])
        pnr.append(res[1])
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    # show summary memmap_list
    # mean_images = []
    # for i,image in enumerate(memmap_list):
    #     mean_images.append(np.mean(image, axis=0))
    #     plt.imshow(mean_images[i])
    #     plt.show()


#%% Estimate minSNR with trough detection
    weighted_matrix = create_weighted_matrix(446, 360)
    cropped_matrix = crop_around_center(weighted_matrix, 418, 446)
    original = cn_filter[0].flatten()
    test = [a*b for a,b in zip(original,cropped_matrix.flatten())]
    plt.hist(test,1000,range = (.1,1))






#%% Calculate correlation image for a certain number  of frames
    # Setup code to load the memap file
    import pickle

    fname_new = r'D:\CaImAn_Data\memmap__d1_418_d2_446_d3_1_order_C_frames_37564.mmap'
    dims: tuple
    Yr, dims, T = cm.load_memmap(fname_new, mode='r+')
    images = Yr.T.reshape((T,) + dims, order='F')
    gSig = (5, 5)
    para_list = [[0,37564,1],[0,37564,10],[0,37564,100],[0,37564,500],[0,37564,1000],[0,10000,5],[0,10000,10],[0,5000,2],[0,5000,5],[0,5000,10],[0,1000,1],[0,1000,2]] # parameter list
    para_list = para_list[::-1]
    times = []          # list to save execution time for the different parameter times[i] = time for para_list[i]
    cn_filter = []      # list of cn_filter for each set of parameters
    pnr = []            # list of pnr_filter for each set of parameters
    for index, para in enumerate(para_list) :
        try:
            st = time.time()
            times.append(0)
            res = cm.summary_images.correlation_pnr(images[para[0]:para[1]:para[2]], gSig=gSig[0], swap_dim=False)
            cn_filter.append(res[0])
            pnr.append(res[1])
            et = time.time()
            times[index] = (et - st)
            print(f'parameters {para_list[index]} took {times[index]} seconds to run')
        except:
            print(f'did not work for para = {index}')
    results_local = {}
    results_local['times'] = times
    results_local['parameters'] = para_list
    with open('results_local.pkl', 'wb') as f:
        pickle.dump(results_local, f)
    # with open('results_local.pkl', 'rb') as f:
    #     result = pickle.load(f)



    # nochmal for the same file from the server
    fname_new = r'O:\archive\projects\2023_students\TestExperiment\data\1_preprocessed\20230405_m1018_wt_1110\miniscope_video\memmap__d1_418_d2_446_d3_1_order_C_frames_37564.mmap'
    dims: tuple
    Yr, dims, T = cm.load_memmap(fname_new, mode='r+')
    images = Yr.T.reshape((T,) + dims, order='F')
    gSig = (5, 5)
    para_list = [[0, 37564, 1], [0, 37564, 10], [0, 37564, 100], [0, 37564, 500], [0, 37564, 1000], [0, 10000, 5],
                 [0, 10000, 10], [0, 5000, 2], [0, 5000, 5], [0, 1000, 1], [0, 1000, 2]]  # parameter list
    times = []  # list to save execution time for the different parameter times[i] = time for para_list[i]
    cn_filter = []  # list of cn_filter for each set of parameters
    pnr = []  # list of pnr_filter for each set of parameters
    for index, para in enumerate(para_list):
        st = time.time()
        res = cm.summary_images.correlation_pnr(images[para[0]:para[1]:para[2]], gSig=gSig[0], swap_dim=False)
        cn_filter.append(res[0])
        pnr.append(res[1])
        et = time.time()
        times.append(et - st)
    result = {}
    result['times'] = times
    result['parameters'] = para_list
    with open('result_from_server.pkl', 'wb') as f:
        pickle.dump(result, f)
    # with open('result_from_server.pkl', 'rb') as f:
    #     result = pickle.load(f)


#%% Prototype code for Python shell
import timer
import logging
import matplotlib.pyplot as plt
import numpy as np
#different matplotlib backend for better plots
import matplotlib
matplotlib.rcParams['backend'] = 'TkAgg'
##
import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
import time
fname_new = r'D:\CaImAn_Data\memmap_d1_340_d2_365_d3_1_order_C_frames_1000.mmap'
Yr, dims, T = cm.load_memmap(fname_new,mode='r+')
images = Yr.T.reshape((T,) + dims, order='F')

gSig = (5,5)
cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=gSig[0], swap_dim=False)

plt.hist(cn_filter.flatten(),100)
plt.hist(pnr.flatten(),2000)
cm.utils.visualization.inspect_correlation_pnr(cn_filter, pnr)
#test = [a+b for a,b in zip(cn_filter.flatten(),cropped_matrix.flatten())]

#%% function to make a weighed matrix that resembles a circle
def create_weighted_matrix(size, sigma):
    center = (size - 1) / 2
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    distance_from_center = np.sqrt((x - center)**2 + (y - center)**2)
    weighted_matrix = np.exp(-0.5 * (distance_from_center / sigma)**2)
    return weighted_matrix

# usage:
# matrix_size = 5  # Adjust the size of the matrix (odd number for symmetry)
# sigma = 1.0      # Adjust the spread of the weights
#
# weighted_matrix = create_weighted_matrix(446, 360)
# print(weighted_matrix)



#%% function to crop a matrix around the center to fit it to certain dimensions
def crop_around_center(matrix, new_rows, new_columns):
    center_row = matrix.shape[0] // 2
    center_col = matrix.shape[1] // 2
    start_row = center_row - new_rows // 2
    start_col = center_col - new_columns // 2
    end_row = start_row + new_rows
    end_col = start_col + new_columns

    cropped_matrix = matrix[start_row:end_row, start_col:end_col]
    return cropped_matrix

#usage :
# new_rows = 3    # Number of rows in the cropped matrix
# new_columns = 2 # Number of columns in the cropped matrix
#cropped_matrix = crop_around_center(weighted_matrix, 418, 446)

