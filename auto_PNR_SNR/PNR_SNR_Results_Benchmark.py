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



    #%% Load Result files
    path = r'O:\archive\projects\2023_students\Result_files\auto_pnr_banchmark\\'  # path to the result files
    auto = list(np.load(path + r'auto.npy',allow_pickle = True))
    auto_mean = list(np.load(path + r'auto_mean.npy', allow_pickle=True))
    auto_grad = list(np.load(path + r'auto_grad.npy',allow_pickle = True))
    manual =  list(np.load(path + r'manual.npy',allow_pickle = True))

    #% Sort the results
    auto_grad.sort(key=lambda x:x.mmap_file ,reverse = True)
    sorted_auto = []
    sorted_mean = []
    sorted_manual = []
    for i in auto_grad:
        for j in range(17):
            if auto[j].mmap_file == i.mmap_file: sorted_auto.append(auto[j])
            if manual[j].mmap_file == i.mmap_file: sorted_manual.append(manual[j])
        for k in range(15):
            if auto_mean[k].mmap_file == i.mmap_file: sorted_mean.append(auto_mean[k])
    sorted_grad = auto_grad

    #%% do post processing to see how many neurons stay

    #DISCARD LOW QUALITY COMPONENTS
    min_SNR = 2.5           # adaptive way to set threshold on the transient size
    r_values_min = 0.85    # threshold on space consistency (if you lower more components
    #                        will be accepted, potentially with worst quality)

    for i in range(15):
        # set paramters
        sorted_grad[i].params.set('quality', {'min_SNR': min_SNR,
                                   'rval_thr': r_values_min,
                                   'use_cnn': False})
        sorted_auto[i].params.set('quality', {'min_SNR': min_SNR,
                                   'rval_thr': r_values_min,
                                   'use_cnn': False})
        sorted_manual[i].params.set('quality', {'min_SNR': min_SNR,
                                   'rval_thr': r_values_min,
                                   'use_cnn': False})
        sorted_mean[i].params.set('quality', {'min_SNR': min_SNR,
                                   'rval_thr': r_values_min,
                                   'use_cnn': False})

        # load the images
        mmap_file = sorted_auto[i].mmap_file
        Yr, dims, T = cm.load_memmap(mmap_file, mode='r')
        images = Yr.T.reshape((T,) + sorted_auto[i].dims, order='F')


        sorted_auto[i].estimates.evaluate_components(images, sorted_auto[i].params, dview=dview)
        print(' ***** ')
        print('Number of total components: ', len(sorted_auto[i].estimates.C))
        print('Number of accepted components: ', len(sorted_auto[i].estimates.idx_components))
        sorted_auto[i].stats['neurons_after_post'] = len(sorted_auto[i].estimates.idx_components)

        sorted_grad[i].estimates.evaluate_components(images, sorted_grad[i].params, dview=dview)
        print(' ***** ')
        print('Number of total components: ', len(sorted_grad[i].estimates.C))
        print('Number of accepted components: ', len(sorted_grad[i].estimates.idx_components))
        sorted_grad[i].stats['neurons_after_post'] = len(sorted_grad[i].estimates.idx_components)

        sorted_manual[i].estimates.evaluate_components(images, sorted_manual[i].params, dview=dview)
        print(' ***** ')
        print('Number of total components: ', len(sorted_manual[i].estimates.C))
        print('Number of accepted components: ', len(sorted_manual[i].estimates.idx_components))
        sorted_manual[i].stats['neurons_after_post'] = len(sorted_manual[i].estimates.idx_components)

        sorted_mean[i].estimates.evaluate_components(images, sorted_mean[i].params, dview=dview)
        print(' ***** ')
        print('Number of total components: ', len(sorted_mean[i].estimates.C))
        print('Number of accepted components: ', len(sorted_mean[i].estimates.idx_components))
        sorted_mean[i].stats['neurons_after_post'] = len(sorted_mean[i].estimates.idx_components)

    #%% pearson correlation
    for i in range(15):
        C1 = sorted_mean[i].estimates.C
        C2 = sorted_grad[i].estimates.C

        # Sum along the temporal axis for each matrix
        summed_activity_C1 = np.sum(C1, axis=1)
        summed_activity_C2 = np.sum(C2, axis=1)

        # Ensure that both summed activities have the same shape
        # This step might be necessary if the number of neurons detected differ between the two matrices
        if summed_activity_C1.shape != summed_activity_C2.shape:
            min_length = min(summed_activity_C1.shape[0], summed_activity_C2.shape[0])
            summed_activity_C1 = summed_activity_C1[:min_length]
            summed_activity_C2 = summed_activity_C2[:min_length]

        # Calculate the Pearson correlation
        correlation_coefficient = np.corrcoef(summed_activity_C1, summed_activity_C2)[0, 1]

        print(f"Pearson correlation coefficient: {correlation_coefficient:.3f}")
