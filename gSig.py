from mesmerize_core import *
import caiman as cm
import numpy as np
from copy import deepcopy
import pandas as pd
import tifffile
from caiman.motion_correction import high_pass_filter_space
from caiman.summary_images import correlation_pnr
from caiman import load_memmap
from pathlib import Path
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf import params as params
from fastplotlib import ImageWidget
from ipywidgets import VBox, IntSlider, Layout


import papermill as pm # to pass parameter to the notebook
import webview
import threading
import os
import time
from myfunctions import runCaiman,getCaimanParams
from copy import deepcopy

params = getCaimanParams()
fnames = [r'C:\Users\ba81\caiman_data\example_movies\1_memmap_d1_340_d2_365_d3_1_order_C_frames_1000.mmap']
params.change_params(params_dict = {'fnames':fnames})

#movie_path = r'C:/Users/ba81/caiman_data/1_memmap_d1_340_d2_365_d3_1_order_C_frames_1000.mmap'

# pass the path to the notebook
pm.execute_notebook(
   r'C:\Users\ba81\mescore_demo.ipynb',
   r'C:\Users\ba81\mescore_demo.ipynb',  # This will be your executed notebook with the passed parameters. You can overwrite the original if desired.
   parameters={'movie_path': str(fnames[0])}
)

# Run the GUI based Mescore Visualization
def run_voila():
    os.system(r'C:\Users\ba81\mescore_demo.ipynb --no-browser')

# Start voila in a separate thread
threading.Thread(target=run_voila).start()

# Give voila a moment to start up
time.sleep(2)

# Create a standalone window with pywebview
webview.create_window("gSigfilt", "http://localhost:8888", width=800, height=650)
webview.start()


gSig_filt = 7
gSig_filt = (gSig_filt,gSig_filt)
params.change_params(params_dict = {'gSig_filt':gSig_filt})
params_list = [deepcopy(params) for i in range(3)]
for idx,i in enumerate(range(-2,3,2)):

    gSig = gSig_filt[0]+i
    gSig = (gSig,gSig)

    gSiz = (4 * gSig[0] + 1)
    gSiz = (gSiz,gSiz)
    params_list[idx].change_params(params_dict={'gSiz': gSiz,
                                                'gSig': gSig})


res = []
# start the cluster
try:
    cm.stop_server()  # stop it if it was running
except():
    pass

c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                 n_processes=24,
                                                 # number of process to use, if you go out of memory try to reduce this one
                                                 single_thread=False)

for par in params_list:

    cnm1 = cnmf.CNMF(n_processes=n_processes,params=opts, dview=dview)
    cnm1.fit_file(motion_correct=True)
    Yr, dims, T = cm.load_memmap(cnm1.mmap_file)
    images = Yr.T.reshape((T,) + dims, order='F')
    cnm1.estimates.evaluate_components(images, cnm1.params)
    res.append(cnm1)



