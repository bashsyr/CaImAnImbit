"""
Code to automate the process of choosing the min_pnr and min_corr parameters
"""
#%% imports
    import matplotlib
    matplotlib.rcParams['backend'] = 'Qt5Agg'  # works for win 11
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
    for path in paths:

        files_names = glob.glob(path + r'C_frames*.mmap')
        files_names = natsort.natsorted(files_names)
        file_name = files_names[-1]
        # opts.change_params(params_dict={'fnames': files_names})
        # runCaiman(opts, parameter_name = 'manual')



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
        ##
        cnm1.save(results_path + parameter_name + "_Time=" + str(int(elapsed_time)) + "s" + "_results.hdf5")
        # print('Execution time:', elapsed_time, 'seconds')



def Calulate_minPNR_minCORR(cn_filters: list = None, pnr_list: list = None):
    if cn_filters == None or pnr_list == None:
        print("no filters provided, using default values (min_corr = .85) , min_pnr = 5.9 ")
        min_corr = .85
        min_pnr = 5.9
        return min_corr, min_pnr
    if len(cn_filters) != len(cn_filters):
        raise Warning('Warning: pnr and cn_cilter lists are not the same size')

    # do trough detection to calculate min_corr
    Troughs = [1 for i in range(len(cn_filters))]
    for i in range(len(cn_filters)):

        data = cn_filters[i].flatten()
        kde = gaussian_kde(data)

        # Define the range for evaluating KDE
        x = np.linspace(.2, max(data), 1000)  # .2 to prevent it from detecting peaks at the beginning
        y = kde.evaluate(x)

        # Find peaks
        peaks, _ = find_peaks(y, distance=20)  # 'distance' may need adjustment based on your data

        # Find the trough between the main peaks (assuming the two main peaks are the first two found)
        if len(peaks) > 1:
            trough = np.argmin(y[peaks[0]:peaks[1]]) + peaks[0]
        else:
            trough = None

        # # Visualize
        # plt.plot(x, y, label='KDE')
        # plt.plot(x[peaks], y[peaks], "x", label='Peaks')
        # if trough is not None:
        #     plt.plot(x[trough], y[trough], "o", label='Trough')
        # plt.legend()
        # plt.show()

        if trough is not None:
            # print(f'Trough value: {x[trough]}')
            Troughs[i] = x[trough]

        # else:
        #     print("Could not identify a clear trough between peaks.")

    Troughs = [round(i, 2) for i in Troughs]
    min_corr = min(Troughs)

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

    return min_corr, min_pnr
