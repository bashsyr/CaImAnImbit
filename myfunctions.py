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





#%% Function to run the pipeline at once
def runCaiman(opts:any,parameter_name: str):
    st = time.time()
    cnm1 = cnmf.CNMF(n_processes=n_processes, params=opts, dview=dview)
    cnm1.fit_file(motion_correct=True)
    del cnm1.estimates.f
    et = time.time()
    elapsed_time = et - st
    # collect stats and put them in stats dictionary
    cnm1.stats = {}
    cnm1.stats['time'] = elapsed_time
    cnm1.stats['Parameters:'] = parameter_name
    cnm1.stats['Number_of_neuros'] = len(cnm1.estimates.C)
    # cnm1.stats['min_pnr'] = cnm1.params
    ##
    cnm1.save(results_path + parameter_name + "_Time=" + str(int(elapsed_time)) + "s" + "_results.hdf5")
    # print('Execution time:', elapsed_time, 'seconds')


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


def calculate_minPNR(pnr_list: list):
    # Do a gradient decen degree detection to calculate min_pnr
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

    min_pnr = round(np.mean(pnrs), 2)
    # visualize
    # # Visualize
    # plt.plot(x, y, label='KDE')
    # plt.plot(x[main_peak_idx], y[main_peak_idx], "x", label='Main Peak')
    # plt.plot(x[closest_idx], y[closest_idx], "o", label='Closest Point to Target Gradient')
    # plt.legend()
    # plt.show()

    return min_pnr

