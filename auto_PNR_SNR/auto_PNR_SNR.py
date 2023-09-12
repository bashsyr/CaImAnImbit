"""
Code to automate the process of choosing the min_pnr and min_corr parameters
"""
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
    import pickle as pkl
    from caiman.source_extraction.cnmf import params as params
    from caiman.source_extraction import cnmf
    import natsort
    import glob
    import caiman as cm
    from pathlib import Path
#%% Load parameters and files


# Load Caiman parameter Dict
    param_file_path = r'O:\archive\projects\2023_students\playground_folder\Caiman_parameters.pkl'
    results_path = r'O:\archive\projects\2023_students\Result_files\auto_pnr_banchmark'
    with open(param_file_path, 'rb') as f:
        parameters = pkl.load(f)
    params_dict = parameters['mc_dict'] | parameters['params_dict']
    default_opts = params.CNMFParams(params_dict=params_dict)
    opts = params.CNMFParams(params_dict=params_dict)
    # load the files
    paths  = [r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230705_m1310_som_1410\miniscope_video\*',
              r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230704_m1310_som_1939\miniscope_video\*',
              r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230627_m1310_som_1325\miniscope_video\*',
              r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230620_m1310_som_1711\miniscope_video\*',
              r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230619_m1310_som_0947\miniscope_video\*',
              r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230616_m1310_som_1554\miniscope_video\*',
              r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230707_m1309_som_1152\miniscope_video\*',
              r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230612_m1309_som_1302\miniscope_video\*',
              r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230707_m1308_som_1520\miniscope_video\*',
              r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230615_m1308_som_0936\miniscope_video\*',
              r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230616_m1310_som_1150\miniscope_video\*',
              r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230705_m1309_som_1453\miniscope_video\*',
              r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230705_m1308_som_1534\miniscope_video\*',
              r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230608_m1308_som_1636\miniscope_video\*',
              r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230704_m1308_som_1813\miniscope_video\*',
              r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230608_m1310_som_1357\miniscope_video\*',
              r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230621_m1310_som_0933\miniscope_video\*',
              r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230802_m1309_som_1746\miniscope_video\*',
              r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230809_m1310_som_1243\miniscope_video\*',
              r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230801_m1310_som_1056\miniscope_video\*',
              r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230728_m1310_som_1747\miniscope_video\*'
              ]



#%% The main Benchmark code
    for index,i in enumerate(paths):
        # run caiman with standard parameters and collect results
        files_names = glob.glob(i + r'*C_frames*.mmap')
        files_names = natsort.natsorted(files_names)
        if not files_names:
            pass
        file_name = files_names[-1]
        opts.change_params(params_dict={'fnames': file_name})

        # runCaiman(opts, parameter_name = f'{index+4}manual')


        # calculate the PNR and SNR and use these values
        # cn_filters,pnr_list = get_cn_pnr(i)
        min_corr_list = [.7]
        min_pnr_list =  [5,5.5,6,6.5,7,7.5]
        for min_corr in min_corr_list:
            for min_pnr in min_pnr_list:
                opts.change_params(params_dict={'min_corr': min_corr,'min_pnr': min_pnr})
                runCaiman(opts, parameter_name=f'{index}min_corr={min_corr}_min_pnr={min_pnr}')

        # # reset to original parameters
        opts.change_params(params_dict=params_dict)



# %% start the cluster
    try:
        cm.stop_server()  # stop it if it was running
    except():
        pass

    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', # if error try to change ('local','ipyparallel')
                                                     n_processes=16,  # number of process to use, if you go out of memory try to reduce this one
                                                     single_thread=False)


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


def calculate_minCORR_with_Gradient_descent(cn_filters: list = None):
    # Do a gradient descent degree detection to calculate min_corr
    corrs = []
    for i in range(len(cn_filters)):
        data = cn_filters[i].flatten()
        kde = gaussian_kde(data)

        # Define the range for evaluating KDE
        x = np.linspace(0.3, max(data), 1000)
        y = kde.evaluate(x)

        # Calculate the gradient
        gradient = np.gradient(y)

        # Target gradient value
        target_gradient = -.005  # define the tan of the slope to get the desired cutt-off

        # Find the index closest to the target gradient after the main peak
        main_peak_idx = np.argmax(y)
        closest_idx = np.argmin(np.abs(gradient[main_peak_idx:] - target_gradient)) + main_peak_idx

        # set min_pnr as the x value of the appropriate index
        corrs.append(x[closest_idx])

        # # Visualize
        plt.figure(i)
        plt.plot(x, y, label='KDE')
        plt.plot(x[main_peak_idx], y[main_peak_idx], "x", label='Main Peak')
        plt.plot(x[closest_idx], y[closest_idx], "o", label='Closest Point to Target Gradient')
        plt.legend()
        plt.show()

    min_corr = round(np.mean(corrs), 2)
    print(corrs)
    return min_corr


def get_cn_pnr (path: str , file_increment = 5, frame_increment = 1,n = 5):
    """

    :param path: parent folder path that contains the F_frame mmap files
    :param file_increment: take every Nth file (ex: file_increment = 5 would take every 5th mmmap)
    :param frame_increment: the caiman implemented frame increment (take every nth frame in a file similar to file increment)
    :param n: Number of cn filters returned (if you choose n = 5 you will get 5 cn_filters back calculated (for file_increment =5)
              from files (1,5,10,15) for the first filter, then (2,6,11,16) for the second and so on
    :return: two arrays
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
        images = []
        st = time.time()
        memmap_list = []  # a list of the individual videos as memmaps
        for index, name in enumerate(files_names):
            Yr, dims, T = cm.load_memmap(name, mode='r')
            memmap_list.append(Yr.T)
        gSig = (5, 5)
        try:
            images = np.concatenate(([item for item in memmap_list[r::file_increment]]), axis=0)
            images = images.reshape(len(images), dims[0], dims[1], order='F')
            cn_filter, pnr = cm.summary_images.correlation_pnr(images[::frame_increment], gSig=gSig[0], swap_dim=False)
        except:
            pass
        et = time.time()
        elapsed_time = et - st
        print('Execution time:', elapsed_time, 'seconds')
        cn_filters.append(cn_filter)
        pnr_list.append(pnr)

    return cn_filters,