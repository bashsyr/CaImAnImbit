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

# %% start the cluster
    try:
        cm.stop_server()  # stop it if it was running
    except():
        pass

    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', # if error try to change
                                                     n_processes=61,  # number of process to use, if you go out of memory try to reduce this one
                                                     single_thread=False)





#%% import Results files
    parameter_name = 'stride='         # if you want to look for results of special parameter eg: 'rf=' the '=' sign to only look for para_name
    path = r'D:\CaImAn_Data\*'          # path to the result files
    n_frames = 1000  # number of frames in the movie
    result_files_names = glob.glob(path + parameter_name + r'*.hdf5')
    cnm = []    # array of ciman objects to load the results
    for name in result_files_names:
        cnm.append(cm.source_extraction.cnmf.cnmf.load_CNMF(name))




#%% plot the results:
    results = cm.base.rois.register_multisession([m.estimates.A for m in cnm], dims = cnm[0].dims)
    fig, axes = plt.subplots(len(results[1]), figsize = (10,30))
    x = np.arange(n_frames)/25
    for ii,ele in enumerate(results[1]):
    #   shared_comps[1][1:5]
        print(ele[0])
        if np.isnan(ele[0]):
            continue
        if np.isnan(ele[1]):
            continue
        for m in cnm:
            axes[ii].plot(x,m.estimates.C[int(ele[0])])


cm.base.rois.find_matches()
