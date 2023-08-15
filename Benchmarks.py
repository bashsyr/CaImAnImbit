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
    # %% start the cluster
    try:
        cm.stop_server()  # stop it if it was running
    except():
        pass

    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', # if error try to change
                                                     n_processes=61,  # number of process to use, if you go out of memory try to reduce this one
                                                     single_thread=False)



    #%%  Standard parameters definition

    # dataset dependent parameters
    fnames = [r'D:\CaImAn_Data\1.avi']  # filename to be processed
    filename_reorder = fnames

    fr = 10                          # movie frame rate
    decay_time = 0.4                 # length of a typical transient in seconds (half?)

    # motion correction parameters
    motion_correct = True            # flag for motion correction
    pw_rigid = False                 # flag for pw-rigid motion correction
    gSig_filt = (3, 3)   # size of filter, in general gSig (see below),
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
    test_opts = params.CNMFParams(params_dict=mc_dict) # write paramiters into a dic and then  pass it to the algorithm



    #  Parameters for source extraction and deconvolution (CNMF-E algorithm)
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
    dims: tuple() = (340, 365)

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
                                    'border_pix': 0})  # number of pixels to not consider in the borders)
    test_opts = opts

    # parameter_name = "Parameter:"


    # %%SET THE PIPELINE FUNCTION TO RUN AT ONCE AND collect some stats
    def runCaiman(parameter_name: str):
        st = time.time()
        cnm1 = cnmf.CNMF(n_processes=61, params=test_opts, dview=dview)
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
        cnm1.save(fnames[0] + parameter_name + "_Time=" + str(int(elapsed_time)) + "s" + "_results.hdf5")
        print('Execution time:', elapsed_time, 'seconds')



#%% Loop through the possible parameters, test them and save the results
    start = time.time()
    def loop():


        parameter_name = r"Parameter_"
        parameter_name_default = r"Parameter_"
        # for t_p in range(0, 3):
        #     parameter_name = parameter_name + 'p=' + str(t_p)
        #     test_opts.change_params(params_dict={'p':t_p})
        #     try:
        #         runCaiman(parameter_name)
        #     except:
        #         print('CaImAn did not work for p='+ str(t_p))
        #     parameter_name = parameter_name_default
        # test_opts.change_params(params_dict={'p': p})

        # for t_rf in range (20,61,5):
        #     parameter_name = parameter_name + 'rf=' + str(t_rf)
        #     test_opts.change_params(params_dict={'rf': t_rf})
        #     try:
        #         runCaiman(parameter_name)
        #     except:
        #         print('CaImAn did not work for rf=' + str(t_rf))
        #     parameter_name = parameter_name_default
        # test_opts.change_params(params_dict={'rf': rf})
        #
        # for t_merge_thr in np.arange (.6,.96,.05):
        #     t_merge_thr = round(t_merge_thr, 2)
        #     parameter_name = parameter_name + 'merge_thr=' + str(t_merge_thr)
        #     test_opts.change_params(params_dict={'merge_thr': t_merge_thr})
        #     runCaiman(parameter_name)
        #     parameter_name = parameter_name_default
        # test_opts.change_params(params_dict={'merge_thr': merge_thr})
        #
        # for t_stride in range (20,50,5):
        #     parameter_name = parameter_name + 'stride=' + str(t_stride)
        #     test_opts.change_params(params_dict={'stride': t_stride})
        #     runCaiman(parameter_name)
        #     parameter_name = parameter_name_default
        # test_opts.change_params(params_dict={'stride': stride_cnmf})



        for t_gnb in range (-2,3,1):
            parameter_name = parameter_name + 'gnb=' + str(t_gnb)
            test_opts.change_params(params_dict={'nb': t_gnb})
            runCaiman(parameter_name)
            parameter_name = parameter_name_default
        test_opts.change_params(params_dict={'nb': gnb})

        for t_decay_time in np.arange (0.2,1.6,0.1):
            t_decay_time = round(t_decay_time, 1)
            parameter_name = parameter_name + 'decay_time=' + str(t_decay_time)
            test_opts.change_params(params_dict={'decay_time':t_decay_time})
            try:
                runCaiman(parameter_name)
            except:
                print('CaImAn did not work for decay_time='+ str(t_decay_time))
            parameter_name = parameter_name_default
        test_opts.change_params(params_dict={'decay_time': decay_time})


        for t_ring_size_factor in np.arange (1.0,2,0.1):
            t_ring_size_factor = round(t_ring_size_factor, 1)
            parameter_name = parameter_name + 'ring_size_factor=' + str(t_ring_size_factor)
            test_opts.change_params(params_dict={'ring_size_factor':t_ring_size_factor})
            try:
                runCaiman(parameter_name)
            except:
                print('CaImAn did not work for ring_size_factor='+ str(t_ring_size_factor))
            parameter_name = parameter_name_default
        test_opts.change_params(params_dict={'ring_size_factor': ring_size_factor})



        for t_K in np.arange (0,6,1):
            t_K = round(t_K, 1)
            parameter_name = parameter_name + 'K=' + str(t_K)
            test_opts.change_params(params_dict={'K':t_K})
            try:
                runCaiman(parameter_name)
            except:
                print('CaImAn did not work for K='+ str(t_K))
            parameter_name = parameter_name_default
        test_opts.change_params(params_dict={'K': K})



        # loop for changing gSig, gSiz at the same time
        for t_gSig in range (2,8,1):
            tuple_gSig = (t_gSig, t_gSig)
            parameter_name = parameter_name + 'gSig=' + str(t_gSig)
            test_opts.change_params(params_dict={'gSig':(t_gSig, t_gSig)})
            for t_gSiz in range (4*t_gSig-4,4*t_gSig+5):
                temp = parameter_name
                parameter_name = parameter_name + 'gSiz=' + str(t_gSiz)
                test_opts.change_params(params_dict={'gSiz': (t_gSiz, t_gSiz)})
                runCaiman(parameter_name)
                parameter_name = temp # to remove gSiz from the name but keep gSig
            test_opts.change_params(params_dict={'gSiz': gSiz})
            parameter_name = parameter_name_default
        test_opts.change_params(params_dict={'gSig': gSig})

    loop()

    endt = time.time()
    elapsed_time = endt - start
    elapsed_time = int(elapsed_time) / 60
    with open(r'D:\CaImAn_Data\time.txt', 'w') as f:
        f.write('%d' % elapsed_time)
    #os.system("shutdown /s /t 1")


