    """
    Code snip to compare and plot different results (of the same animal) to check the effect of changing certain parameters
    Author: Bashar
    """

    # CAIMAN imports
    import matplotlib.pyplot as plt
    import numpy as np
    #different matplotlib backend for better plots
    import matplotlib
    matplotlib.rcParams['backend'] = 'Qt5Agg'  # works for win 11

    # Hotfix for win 11 plot issue
    #
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'qt')
    ##
    import caiman as cm
    from caiman.source_extraction import cnmf
    from caiman.utils.utils import download_demo
    from caiman.utils.visualization import inspect_correlation_pnr
    from caiman.motion_correction import MotionCorrect
    from caiman.source_extraction.cnmf import params as params
    # extra imports for diverse code parts
    import time
    import os
    import contextlib
    import io
    import glob
    import scipy
    import natsort
    import pandas as pd
    import seaborn as sns
    from numpy import mean as avg

# %% start the cluster
    try:
        cm.stop_server()  # stop it if it was running
    except():
        pass

    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', # if error try to change
                                                     n_processes=16,  # number of process to use, if you go out of memory try to reduce this one
                                                     single_thread=False)





#%% import Results files
    parameter_name = ''         # if you want to look for results of special parameter eg: 'rf=' the '=' sign to only look for para_name
    # path = r'O:\archive\projects\2023_students\Result_files\first_Benchmark_to_establish_parameter_baseline\*'
    path = r'C:\Users\ba81\caiman_data\results\florian_final\*'          # path to the result files
    # path = r'O:\archive\projects\2023_students\mesmerize\mesmerize-cnmfe\batch_10\results\florian_linear\*'
    # path = r'O:\archive\projects\2023_students\mesmerize\mesmerize-cnmfe\batch_99\results\*'
    result_files_names = glob.glob(path + parameter_name + r'*.hdf5')
    # result_files_names = result_files_names[:100:]
    result_files_names = natsort.natsorted(result_files_names)
    cnm = []    # array of ciman objects to load the results
    for name in result_files_names[5000::]:
        cnm.append(cm.source_extraction.cnmf.cnmf.load_CNMF(name))
    for cnm1 in cnm:
        cnm1.stats = {}
        cnm1.stats['Number_of_neurons'] = len(cnm1.estimates.C)
        cnm1.stats['Number_of_accepted_neurons'] = len(cnm1.estimates.idx_components)
        cnm1.stats['gSig'] = cnm1.params.init['gSig']
        cnm1.stats['min_corr'] = cnm1.params.init['min_corr']
        cnm1.stats['min_pnr'] = cnm1.params.init['min_pnr']
        cnm1.stats['merge_thr'] = (cnm1.params.merging['merge_thr'])
        cnm1.stats['summed_activity'] = (np.sum(cnm1.estimates.C, axis=0))
        cnm1.stats['K'] = cnm1.params.init['K']
        if cnm1.stats['K'] == None: cnm1.stats['K'] = 0
        # define a quiality metric as a list with the first component (0,1) averaging the r values (spatial consistency) and cnn values, and the second one is average SNR value
        cnm1.stats['average_quality'] = [(avg(cnm1.estimates.r_values) + avg(cnm1.estimates.cnn_preds)) / 2, avg(cnm1.estimates.SNR_comp)]
        cnm1.stats['ring_size_factor'] = cnm1.params.init['ring_size_factor']
    #%% sort the result files
    sorted_cnm  = sorted(cnm, key=lambda x: (x.stats['gSig'][0],x.stats['min_corr'],x.stats['min_pnr'],x.stats['merge_thr'],x.stats['ring_size_factor'],x.stats['K']))



#%% prepare data to plot
    Number_of_neurons = []
    Number_of_accepted_neurons = []
    gSig = []
    min_corr = []
    min_pnr = []
    merge_thr = []
    summed_activity = []
    K = []
    average_quality = []
    ring_size_factor = []
    for cnm1 in sorted_cnm:
        Number_of_neurons.append(cnm1.stats['Number_of_neurons'])
        Number_of_accepted_neurons.append(cnm1.stats['Number_of_accepted_neurons'])
        gSig.append(cnm1.stats['gSig'][0])
        min_corr.append(cnm1.stats['min_corr'])
        min_pnr.append(cnm1.stats['min_pnr'])
        merge_thr.append(cnm1.stats['merge_thr'])
        K.append(cnm1.stats['K'])
        summed_activity.append(cnm1.stats['summed_activity'])
        average_quality.append(cnm1.stats['average_quality'])
        ring_size_factor.append(cnm1.stats['ring_size_factor'])

    results = {
        'Number_of_neurons':Number_of_neurons,
        'Number_of_accepted_neurons':Number_of_accepted_neurons,
        'gSig':gSig,
        'min_corr':min_corr,
        'min_pnr':min_pnr,
        'merge_thr':merge_thr,
        'ring_size_factor':ring_size_factor,
        'average_quality':average_quality,
        'K':K,
        'summed_activity':summed_activity

    }

    res = pd.DataFrame(results)
    df = pd.concat([df, res], ignore_index=True)

    df.to_pickle(r'O:\archive\projects\2023_students\Result_files\grid_search\florian_final.pkl')
    # df1 = pd.read_pickle(r'O:\archive\projects\2023_students\Result_files\grid_search\megan.pkl')    # to load it

    #%% show some basic stats
    for i,n in enumerate(sorted_cnm):
        print(i,n.stats['K'],n.stats['min_pnr'], n.stats['merge_thr'],n.stats['gSig'][0])



    # %% plt pearson corrolation
    x = []
    y = []
    for i, n in enumerate(sorted_cnm):
        if(n.stats['K'] == 0 and n.stats['min_corr'] == .8  and n.stats['merge_thr'] == 0.8 and n.stats['gSig'][0] == 5):
            print(i,n.stats['min_pnr'])
            C1 = sorted_cnm[1].estimates.C
            C2 = n.estimates.C
            y.append(pearson_correlation(C1,C2))
            x.append(n.stats['min_pnr'])

    plt.plot(x, y)




    #%% pickle the whole list
    with open(r'O:\archive\projects\2023_students\Result_files\grid_search\florain_extended_cnm', 'wb') as f:
          pickle.dump(sorted_cnm, f)
    #
    # #%%plot the data
    # plt.figure(figsize=(12, 6))
    # sns.violinplot(data=df, x='gSig', y='Number_of_neurons', hue='min_corr', split=True)
    # plt.title('Neuron Detection based on Parameters')
    # plt.show()
    #
    #


#
#     #%% plot Temporal results:
#     results = cm.base.rois.register_multisession([m.estimates.A for m in cnm], dims = cnm[0].dims)
#     fig, axes = plt.subplots(len(results[1]), figsize = (10,35))
#     n_frames = len(cnm[0].estimates.YrA[0])
#     x = np.arange(n_frames)/25
#     for ii,ele in enumerate(results[1]):
#         if np.isnan(ele[0]):
#             continue
#         if np.isnan(ele[1]):
#             continue
#         for m in cnm:
#             axes[ii].plot(x,m.estimates.C[int(ele[0])])
#
#
#
# #%% plot the shared Neurons across all the sessions
#     common_neurons = [n for n, ele in enumerate(results[1]) if not any(np.isnan(ele))] # list of the shared neurons among all sessions
#     common_A = results[0]               # copy the result matrix to remove the non-shared neurons
#     rest_neurons = [k for k in range (0, len(results[0][0])) if k not in common_neurons]     # get indexes of non-common neurons
#     common_A = np.delete(common_A, rest_neurons, 1) # remove them from the result list
#     # plot the results
#     common_A = scipy.sparse.csc_matrix(common_A)
#     img = np.reshape(np.array(common_A.mean(1)), cnm[1].estimates.dims, order='F')
#     coordinates = cm.utils.visualization.get_contours(common_A, img.shape, thr=0.2, thr_method='max')
#     plt.figure()
#
#     cm.utils.visualization.plot_contours(common_A, img, coordinates=coordinates,
#                                          display_numbers=True,
#                                          cmap='viridis')
#     print("Number of shared Neurons=", len(common_neurons))
#
#


#%% print runs of the ring_size_factor
    for i, n in enumerate(sorted_cnm):
        if(n.stats['K'] == 0 and n.stats['min_corr'] == .8 and n.stats['min_pnr'] == 7 and n.stats['merge_thr'] == 0.8 and n.stats['gSig'][0] == 5):
            print(i, n.stats['ring_size_factor'])
