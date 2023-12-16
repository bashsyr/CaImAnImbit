#!/usr/bin/env python

"""
Complete pipeline for motion correction, source extraction, and deconvolution
of one photon microendoscopic calcium imaging data using the CaImAn package.
The demo demonstrates how to use the params, MotionCorrect and cnmf objects
for processing 1p microendoscopic data. The analysis pipeline is similar as in
the case of 2p data processing with core difference being the usage of the
CNMF-E algorithm for source extraction (as opposed to plain CNMF). Check
the companion paper for more details.

You can also run a large part of the pipeline with a single method
(cnmf.fit_file) See inside for details.

Demo is also available as a jupyter notebook (see demo_pipeline_cnmfE.ipynb)
"""
import timer
import logging
import matplotlib.pyplot as plt
import numpy as np
#different matplotlib backend for better plots
import matplotlib
matplotlib.rcParams['backend'] = 'TkAgg'
##
try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
import time
#%%
# Set up the logger; change this if you like.
# You can log to a file using the filename parameter, or make the output more or less
# verbose by setting level to logging.DEBUG, logging.INFO, logging.WARNING, or logging.ERROR

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.WARNING)
    # filename="/tmp/caiman.log"

#%%

def main():
    pass # For compatibility between running under Spyder and the CLI

# %% start the cluster
    try:
        cm.stop_server()  # stop it if it was running
    except():
        pass

    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', # if error try to change
                                                     n_processes=16,  # number of process to use, if you go out of memory try to reduce this one
                                                     single_thread=False)

# %% First setup some parameters for motion correction
    # dataset dependent parameters
    fnames = [r'O:\archive\projects\2023_students\playground_folder\1.avi']  # filename to be processed
    #fnames = [download_demo(fnames[0])]  # download file if not already present
    filename_reorder = fnames
    fr = 10                          # movie frame rate
    decay_time = 0.4                 # length of a typical transient in seconds (half?)

    # motion correction parameters
    motion_correct = True            # flag for motion correction
    pw_rigid = False                 # flag for pw-rigid motion correction

    gSig_filt = (5, 5)   # size of filter, in general gSig (see below),
    #                      change this one if algorithm does not work (very important para)
    max_shifts = (5, 5)  # maximum allowed rigid shift
    strides = (48, 48)   # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)  # overlap between patches (size of patch strides+overlaps)
    # maximum deviation allowed for patch with respect to rigid shifts
    max_deviation_rigid = 3
    border_nan = 'copy'

    mc_dict = {
        'fnames': fnames,
        'fr': fr,
        'decay_time': decay_time,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'gSig_filt': gSig_filt,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': border_nan
    }

    opts = params.CNMFParams(params_dict=mc_dict) # write paramiters into a dic and then  pass it to the algorithm

# %% MOTION CORRECTION
#  The pw_rigid flag set above, determines where to use rigid or pw-rigid
#  motion correction
    st = time.time()

    if motion_correct:
        # do motion correction rigid
        mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
        mc.motion_correct(save_movie=True)
        fname_mc = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig
        if pw_rigid:
            bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                         np.max(np.abs(mc.y_shifts_els)))).astype(int)
        else:
            bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(int)
            plt.subplot(1, 2, 1); plt.imshow(mc.total_template_rig)  # % plot template
            plt.subplot(1, 2, 2); plt.plot(mc.shifts_rig)  # % plot rigid shifts
            plt.legend(['x shifts', 'y shifts'])
            plt.xlabel('frames')
            plt.ylabel('pixels')

        bord_px = 0 if border_nan == 'copy' else bord_px
        fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C',
                                   border_to_0=bord_px)
    else:  # if no motion correction just memory map the file
        fname_new = cm.save_memmap(filename_reorder, base_name='memmap_',
                                   order='C', border_to_0=0, dview=dview)

    # load memory mappable file
    dims: tuple
    Yr, dims, T = cm.load_memmap(fname_new,mode='r+')
    images = Yr.T.reshape((T,) + dims, order='F')

    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    # test code to subtract the minimum image
    #path_to_normalized_images = 'C:\\Users\\student\\caiman_data\\example_movies\\normalized.mmmap'
    # normalized_images = np.memmap(path_to_normalized_images, dtype=memmap_list.dtype,
    #                               shape=memmap_list.shape, mode='w+')
    # normalized_images = np.clip(normalized_images, 0, 255).astype(np.uint8)

    # fname_normalized = cm.save_memmap(filename_reorder, base_name='normalized_memmap_',
    #                            order='C', border_to_0=0, dview=dview)
    # normalized_Yr, dims, T = cm.load_memmap(fname_normalized,mode='r+')
    # normalized_images = normalized_Yr.T.reshape((T,) + dims, order='F')
    #
#%% Vignette removal


    st = time.time()

    mean_image_before = np.mean(images[:10000:5], axis=0)
    min_image = np.min(images, axis=0)
    images[:] = images - min_image
    mean_image_after = np.mean(images[:10000:5], axis=0)
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    #plot both memmap_list to compare
    fig = plt.figure(figsize=(10, 5))
    # Adds a subplot at the 1st position
    fig.add_subplot(1, 2, 1)

    # showing image
    plt.imshow(mean_image_before)
    plt.axis('off')
    plt.title("before")

    # Adds a subplot at the 2nd position
    fig.add_subplot(1, 2, 2)

    # showing image
    plt.imshow(mean_image_after)
    plt.axis('off')
    plt.title("after")

# %% Parameters for source extraction and deconvolution (CNMF-E algorithm)

    p = 1               # order of the autoregressive system( transients in your data rise instantaneously)
    K = None            # upper bound on number of components per patch, in general None for 1p data
    gSig = (5, 5)       # gaussian width of a 2D gaussian kernel, which approximates a neuron (important!)
    gSiz = (21, 21)     # average diameter of a neuron, in general 4*gSig+1 (important!)
    #gSig = (4, 4)
    #gSiz = (17, 17)
    Ain = None          # possibility to seed with predetermined binary masks
    merge_thr = .7      # merging threshold, max correlation allowed
    rf =30             # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
    #rf = 20
    stride_cnmf = 30    # amount of overlap between the patches in pixels (to prevent splitting cells)
    #                     (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
    tsub = 1            # downsampling factor in time for initialization,
    #                     increase if you have memory problems
    ssub = 2            # downsampling factor in space for initialization,
    #                     increase if you have memory problems
    #                     you can pass them here as boolean vectors
    low_rank_background = None  # None leaves background of each patch intact,
    #                     True performs global low-rank approximation if gnb>0
    gnb = -1             # number of background components (rank) if positive,
    #                     else exact ring model with following settings
    #                         gnb= 0: Return background as b and W
    #                         gnb=-1: Return full rank background B
    #                         gnb<-1: Don't return background
    nb_patch = 0        # number of background components (rank) per patch if gnb>0,
    #                     else it is set automatically
    min_corr = .85       # min peak value from correlation image (important!)
    min_pnr = 5.9        # min peak to noise ration from PNR image (important!)
    ssub_B = 2          # additional downsampling factor in space for background
    ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor (something about the neuro size?)

    opts.change_params(params_dict={'dims': dims,
                                    'method_init': 'corr_pnr',  # use this for 1 photon (use seeding instead of pnr ?)
                                    'K': K,
                                    'gSig': gSig,
                                    'gSiz': gSiz,
                                    'merge_thr': merge_thr,
                                    'p': p,
                                    'tsub': tsub,
                                    'ssub': ssub,
                                    'rf': rf,
                                    'stride': stride_cnmf,
                                    'only_init': True,    # set it to True to run CNMF-E
                                    'nb': gnb,
                                    'nb_patch': nb_patch,
                                    'method_deconvolution': 'oasis',       # could use 'cvxpy' alternatively (back filtering from calcium signal to action potential)
                                    'low_rank_background': low_rank_background,
                                    'update_background_components': True,  # sometimes setting to False improve the results (explanation in minian?)
                                    'min_corr': min_corr,
                                    'min_pnr': min_pnr,
                                    'normalize_init': False,               # just leave as is
                                    'center_psf': True,                    # leave as is for 1 photon
                                    'ssub_B': ssub_B,
                                    'ring_size_factor': ring_size_factor,
                                    'del_duplicates': True,                # whether to remove duplicates from initialization
                                    'border_pix': bord_px})                # number of pixels to not consider in the borders)
# %% Compute a summary image
# change swap dim if output looks weird, it is a problem with tiffile
    cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=gSig[0], swap_dim=False)
    # if your memmap_list file is too long this computation will take unnecessarily
    # long time and consume a lot of memory. Consider changing memmap_list[::1] to
    # memmap_list[::5] or something similar to compute on a subset of the data

    # inspect the summary memmap_list and set the parameters
    inspect_correlation_pnr(cn_filter, pnr)
    # print parameters set above, modify them if necessary based on summary memmap_list
    print(min_corr) # min correlation of peak (from correlation image)
    print(min_pnr)  # min peak to noise ratio


# %% RUN CNMF ON PATCHES

    st = time.time()
    cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=Ain, params=opts)
    cnm.fit(images)
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

# %% ALTERNATE WAY TO RUN THE PIPELINE AT ONCE
    #   you can also perform the motion correction plus cnmf fitting steps
    #   simultaneously after defining your parameters object using
#    cnm1 = cnmf.CNMF(n_processes, params=opts, dview=dview)
#    cnm1.fit_file(motion_correct=True)

# %% DISCARD LOW QUALITY COMPONENTS
    min_SNR = 2.5           # adaptive way to set threshold on the transient size
    r_values_min = 0.85    # threshold on space consistency (if you lower more components
    #                        will be accepted, potentially with worst quality)
    cnm.params.set('quality', {'min_SNR': min_SNR,
                               'rval_thr': r_values_min,
                               'use_cnn': False})
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

    print(' ***** ')
    print('Number of total components: ', len(cnm.estimates.C))
    print('Number of accepted components: ', len(cnm.estimates.idx_components))

# %% PLOT COMPONENTS
    cnm.dims = dims
    display_images = True           # Set to true to show movies and memmap_list
    if display_images:
        cnm.estimates.plot_contours(img=cn_filter, idx=cnm.estimates.idx_components)
        cnm.estimates.view_components(images, idx=cnm.estimates.idx_components)

# %% MOVIES
    display_images = True           # Set to true to show movies and memmap_list
    if display_images:
        # fully reconstructed movie
        cnm.estimates.play_movie(images, q_max=99.5, magnification=2,
                                 include_bck=True, gain_res=10, bpx=bord_px)
        # movie without background
        cnm.estimates.play_movie(images, q_max=99.9, magnification=2,
                                 include_bck=False, gain_res=4, bpx=bord_px)

# %% STOP SERVER
    cm.stop_server(dview=dview)

# %% This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
'''