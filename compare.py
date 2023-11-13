"""
Code snip to compare and plot different results (of the same animal) to check the effect of changing certain parameters
Author: Bashar
"""

# CAIMAN imports
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
# extra imports for diverse code parts
import time
import os
import contextlib
import io
import glob
import scipy
import natsort

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
    path = r'O:\archive\projects\2023_students\mesmerize\mesmerize-cnmfe\batch_1\results\*'          # path to the result files
    result_files_names = glob.glob(path + parameter_name + r'*.hdf5')
    # result_files_names = result_files_names[:100:]
    result_files_names = natsort.natsorted(result_files_names)
    cnm = []    # array of ciman objects to load the results
    for name in result_files_names:
        cnm.append(cm.source_extraction.cnmf.cnmf.load_CNMF(name))
    for cnm1 in cnm:
        cnm1.stats = {}
        cnm1.stats['Number_of_neuros'] = len(cnm1.estimates.C)
        cnm1.stats['Number_of_accepted_neuros'] = len(cnm1.estimates.idx_components)
        cnm1.stats['gSig'] = cnm1.params.init['gSig']
        cnm1.stats['min_corr'] = cnm1.params.init['min_corr']
        cnm1.stats['min_pnr'] = cnm1.params.init['min_pnr']
        cnm1.stats['merge_thr'] = (cnm1.params.merging['merge_thr'])
        cnm1.stats['K'] = cnm1.params.init['K']
        if cnm1.stats['K'] == None: cnm1.stats['K'] = 0

#%% sort the result files
    sorted_cnm  = sorted(cnm, key=lambda x: (x.stats['gSig'][0],x.stats['min_corr'],x.stats['min_pnr'],x.stats['merge_thr'],x.stats['K']))

#%% show basic stats
    for i, n in enumerate(sorted_cnm):
        print(i, n.stats)

    #%% plot Temporal results:
    results = cm.base.rois.register_multisession([m.estimates.A for m in cnm], dims = cnm[0].dims)
    fig, axes = plt.subplots(len(results[1]), figsize = (10,35))
    n_frames = len(cnm[0].estimates.YrA[0])
    x = np.arange(n_frames)/25
    for ii,ele in enumerate(results[1]):
        if np.isnan(ele[0]):
            continue
        if np.isnan(ele[1]):
            continue
        for m in cnm:
            axes[ii].plot(x,m.estimates.C[int(ele[0])])



#%% plot the shared Neurons across all the sessions
    common_neurons = [n for n, ele in enumerate(results[1]) if not any(np.isnan(ele))] # list of the shared neurons among all sessions
    common_A = results[0]               # copy the result matrix to remove the non-shared neurons
    rest_neurons = [k for k in range (0, len(results[0][0])) if k not in common_neurons]     # get indexes of non-common neurons
    common_A = np.delete(common_A, rest_neurons, 1) # remove them from the result list
    # plot the results
    common_A = scipy.sparse.csc_matrix(common_A)
    img = np.reshape(np.array(common_A.mean(1)), cnm[1].estimates.dims, order='F')
    coordinates = cm.utils.visualization.get_contours(common_A, img.shape, thr=0.2, thr_method='max')
    plt.figure()

    cm.utils.visualization.plot_contours(common_A, img, coordinates=coordinates,
                                         display_numbers=True,
                                         cmap='viridis')
    print("Number of shared Neurons=", len(common_neurons))


