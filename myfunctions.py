#%% imports
import matplotlib

matplotlib.rcParams['backend'] = 'Qt5Agg'  # works for win 11
# Hotfix for win 11 plot issue
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'qt')
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
import time
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf import params as params
from pathlib import Path
import natsort
import caiman as cm
import pickle as pkl
import time
def summary_image_from_f_frames(path: str , file_increment = 5, frame_increment = 1,gSig = 5,n = 5):
    """
    :param path: parent folder path that contains the F_frame mmap files
    :param file_increment: take every Nth file (ex: file_increment = 5 would take every 5th mmmap)
    :param frame_increment: the caiman implemented frame increment (take every nth frame in a file similar to file increment)
    :param n: Number of cn filters returned (if you choose n = 5 you will get 5 cn_filters back calculated (for file_increment =5)
              from files (1,5,10,15) for the first filter, then (2,6,11,16) for the second and so on
    :return: two lists
    cn_filters: a list of the filters
    pnr_list: list of pnr arrays
    """
    # calculate the PNR and SNR and use these values
    path = Path(path)
    files_names = list(path.glob(r'*F_frames*1000*.mmap'))
    files_names = natsort.natsorted(files_names)
    if not files_names:
        raise Exception ("empty directoy or wrong path. No F_frames mmap files found")
    cn_filters = []
    pnr_list = []
    for r in range(0, n):

        # images = []
        st = time.time()
        # memmap_list = []  # a list of the individual videos as memmaps
        # for index, name in enumerate(files_names):
        #     Yr, dims, T = cm.load_memmap(name, mode='r')
        #     memmap_list.append(Yr.T)
        gSig = (gSig, gSig)
        images = cm.load_movie_chain(files_names[r::file_increment])
        try:
            # images = np.concatenate(([item for item in memmap_list[r::file_increment]]), axis=0)
            # images = images.reshape(len(images), dims[0], dims[1], order='F')
            cn_filter, pnr = cm.summary_images.correlation_pnr(images[::frame_increment], gSig=gSig[0], swap_dim=False)
        except:
            pass
        et = time.time()
        elapsed_time = et - st
        print('Execution time:', elapsed_time, 'seconds')


        cn_filters.append(cn_filter)
        pnr_list.append(pnr)

    return cn_filters, pnr_list

# %% pearson correlation
def pearson_correlation(C1, C2):

    # Sum along the temporal axis for each matrix
    summed_activity_C1 = np.sum(C1, axis=0)
    summed_activity_C2 = np.sum(C2, axis=0)

    # Ensure that both summed activities have the same shape
    # This step might be necessary if the number of neurons detected differ between the two matrices
    if summed_activity_C1.shape != summed_activity_C2.shape:
        min_length = min(summed_activity_C1.shape[0], summed_activity_C2.shape[0])
        summed_activity_C1 = summed_activity_C1[:min_length]
        summed_activity_C2 = summed_activity_C2[:min_length]

    # Calculate the Pearson correlation
    correlation_coefficient = np.corrcoef(summed_activity_C1, summed_activity_C2)[0, 1]

    print(f"Pearson correlation coefficient: {correlation_coefficient:.3f}")
    return round(correlation_coefficient,2)


# Load Caiman parameter Dict
def getCaimanParams():
    param_file_path = r'O:\archive\projects\2023_students\playground_folder\Caiman_parameters.pkl'
    with open(param_file_path, 'rb') as f:
        parameters = pkl.load(f)
    params_dict = parameters['mc_dict'] | parameters['params_dict']
    opts = params.CNMFParams(params_dict=params_dict)
    return opts


#%% Function to run the pipeline at once
def runCaiman(opts:any,parameter_name: str,save = False, results_path = '' ):
    """
    function to run the pipeline at once and save the results, recomended flow: call opts = getCaimanParams() to get
    the standard parameter set, add the file(s) name(s) of the files zou want to process and then run the  runCaiman(opts, parameter_name)

    :param opts: Caiman parameter set
    :param parameter_name: name of the parameter your're trying to optimize for (going to be added to the result file name)
    :param save: flag for saving the result
    :param results_path: result file path
    :return:
    """
    st = time.time()

    #start the cluster
    try:
        cm.stop_server()  # stop it if it was running
    except():
        pass

    c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                     n_processes=24,
                                                     # number of process to use, if you go out of memory try to reduce this one
                                                     single_thread=False)

    cnm1 = cnmf.CNMF(n_processes=n_processes, params=opts, dview=dview)
    cnm1.fit_file(motion_correct=True)
    Yr, dims, T = cm.load_memmap(cnm1.mmap_file)
    images = Yr.T.reshape((T,) + dims, order='F')
    cnm1.estimates.evaluate_components(images, cnm1.params)
    del cnm1.estimates.f
    et = time.time()
    elapsed_time = et - st
    # collect stats and put them in stats dictionary
    cnm1.stats = {}
    cnm1.stats['time'] = elapsed_time
    cnm1.stats['Parameters:'] = parameter_name
    cnm1.stats['Number_of_neuros'] = len(cnm1.estimates.C)
    cnm1.stats['min_pnr'] = cnm1.params.init['min_pnr']
    cnm1.stats['min_corr'] = cnm1.params.init['min_corr']
    print('Execution time:', elapsed_time, 'seconds')
    ##
    if (save):
        cnm1.save(results_path + parameter_name + "_Time=" + str(int(elapsed_time)) + "s" + "_results.hdf5")

    #stop the server
    try:
        cm.stop_server()  # stop it if it was running
    except():
        pass

    return  cnm1

def calculate_minCORR(cn_filters: list):
    # do trough detection to calculate min_corr
    Troughs = []
    for i in range(len(cn_filters)):

        data = cn_filters[i].flatten()
        kde = gaussian_kde(data)

        # Define the range for evaluating KDE
        x = np.linspace(min(data), max(data), 1000)
        y = kde.evaluate(x)

        # Find peaks
        peaks, _ = find_peaks(y, distance=20)  # 'distance' may need adjustment based on your data

        # Find the trough between the main peaks (highest one and last one)
        if len(peaks) > 1:
            highest_peak_idx = np.argmax(y[peaks])
            if(peaks[highest_peak_idx] != peaks[-1]): # if the highest peak is the last then pass( because no second peak was detected)
                trough = np.argmin(y[peaks[highest_peak_idx]:peaks[-1]]) + peaks[highest_peak_idx]
                if y[trough] + .02 > y[peaks[-1]] : # check if the trough is lower than the second peak (prevents fake troughs)
                    trough = None
            else:
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


    Troughs = [round(i, 2) for i in Troughs]
    print(Troughs)
    if Troughs:
        min_corr = np.mean(Troughs)
    else:
        print('Auto min_corr selection failed, set to default value [min_corr = .85] and flag for human inspection')
        min_corr = .85
    return min_corr


def calculate_minPNR(pnr_list: list):
    # Do a gradient descent degree detection to calculate min_pnr
    pnrs = []
    for i in range(len(pnr_list)):
        data = pnr_list[i].flatten()
        kde = gaussian_kde(data)

        # Define the range for evaluating KDE
        x = np.linspace(min(data), max(data), 1000)
        y = kde.evaluate(x)

        # Calculate the gradient
        gradient = np.gradient(y)

        # Target gradient value
        target_gradient = -.0025  # define the tan of the slope to get the desired cutt-off

        # Find the index closest to the target gradient after the main peak
        main_peak_idx = np.argmax(y)
        closest_idx = np.argmin(np.abs(gradient[main_peak_idx:] - target_gradient)) + main_peak_idx

        # set min_pnr as the x value of the apropriate index
        pnrs.append(x[closest_idx])

        # Visualize
        plt.figure(i)
        plt.plot(x, y, label='KDE')
        plt.plot(x[main_peak_idx], y[main_peak_idx], "x", label='Main Peak')
        plt.plot(x[closest_idx], y[closest_idx], "o", label='Closest Point to Target Gradient')
        plt.legend()
        plt.show()

    min_pnr = round(np.mean(pnrs), 2)



    return min_pnr

