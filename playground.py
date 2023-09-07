#%% Imports
import seaborn
import matplotlib.pyplot as plt
    import numpy as np

    #different matplotlib backend for better plots
    import matplotlib
    matplotlib.rcParams['backend'] = 'TkAgg'
    ##
    import caiman as cm
    #from caiman.source_extraction import cnmf
    from caiman.utils.utils import download_demo
    from caiman.utils.visualization import inspect_correlation_pnr
    #from caiman.motion_correction import MotionCorrect
    #from caiman.source_extraction.cnmf import params as params

    # Other imports
    import glob
    import time
    import natsort
    import pickle as pkl


#%% Test for the automatic PNR CORR calculation

    results = {}
    times = []
    path_list = []
    cn_filters = []
    pnr_list = []
    frame_increments = []


    results_file = r'O:\archive\projects\2023_students\Result_files\Test_summary_image_result.pkl'


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


    for path in paths:
        files_names = glob.glob(path + r'*F_frames*1000.mmap')
        files_names = natsort.natsorted(files_names)
        for r in range(0,5):
            file_increment = 5
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
                cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=gSig[0], swap_dim=False)
            except:
                pass
            et = time.time()
            elapsed_time = et - st
            print('Execution time:', elapsed_time, 'seconds')
            file_increment = str(file_increment)+ '_' + str(r)

            frame_increments.append(file_increment)
            times.append(elapsed_time)
            cn_filters.append(cn_filter)
            pnr_list.append(pnr)
            path_list.append(path)
            # set for the next round
            del images
            del memmap_list

    # Save results in file
    results['frame_increments'] = frame_increments
    results['times'] = times
    results['paths'] = path_list
    results['cn_filters'] = cn_filters
    results['pnr_list'] = pnr_list
    pkl.dump(results, open(results_file, "wb"))



    #
    # results = pkl.load(open(results_file, "rb"))
    # times = results['times']
    # path_list =  results['paths']
    # cn_filters =  results['cn_filters']
    # pnr_list =  results['pnr_list']
    # frame_increments = results['frame_increments']


























#%% test code to make the DB calulate Summary image on a seet of F_ordered memmaps instead of the big C_ordered one
    st = time.time()
    # path = r'D:\CaImAn_Data\data\1_preprocessed\20230405_m1018_wt_1110\*'          # path to the result files
    path = r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230405_m1018_wt_1110\miniscope_video\*'         # sever path
    frame_increment = 5         # downsamppling parameter
    files_names = glob.glob(path + r'*F_frames*1000.mmap')
    files_names = natsort.natsorted(files_names)
    memmap_list = []    # a list of the individual videos as memmaps
    for index,name in enumerate(files_names):
        Yr, dims, T = cm.load_memmap(name, mode='r')
        memmap_list.append(Yr.T)
    gSig = (5, 5)
    images = np.concatenate(([item for item in memmap_list[::frame_increment]]),axis = 0)
    images = images.reshape(len(images),dims[0],dims[1],order='F')
    cn_filter,pnr = cm.summary_images.correlation_pnr(images[::2], gSig=gSig[0], swap_dim=False)
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    # cm.utils.visualization.inspect_correlation_pnr(cn_filter, pnr)



#%% TEST the test code for Summary image
    results = {}
    times = []
    frame_increments = []
    cn_filters = []
    pnr_list = []

    # path = r'D:\CaImAn_Data\data\1_preprocessed\20230405_m1018_wt_1110\*'          # path to the result files
    path = r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230405_m1018_wt_1110\miniscope_video\*'  # sever path
    files_names = glob.glob(path + r'*F_frames*1000.mmap')
    files_names = natsort.natsorted(files_names)


    for frame_increment in range(10,0,-1):
        images = []
        st = time.time()
        memmap_list = []  # a list of the individual videos as memmaps
        for index, name in enumerate(files_names):
            Yr, dims, T = cm.load_memmap(name, mode='r')
            memmap_list.append(Yr.T)
        gSig = (5, 5)
        try:
            images = np.concatenate(([item for item in memmap_list[::frame_increment]]), axis=0)
            images = images.reshape(len(images), dims[0], dims[1], order='F')
            cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=gSig[0], swap_dim=False)
        except:
            pass
        et = time.time()
        elapsed_time = et - st
        print('Execution time:', elapsed_time, 'seconds')

        # #collect results
        # results = pkl.load(open("ResultsDB_F_Frame_Files.p", "rb"))
        # times = results['times']
        # frame_increments =  results['frame_increments']
        # cn_filters =  results['cn_filters']
        # pnr_list =  results['pnr_list']

        times.append(elapsed_time)
        cn_filters.append(cn_filter)
        pnr_list.append(pnr)
        frame_increments.append(frame_increment)

        # set for the next round
        del images
        del memmap_list
    # do a safety save before testing the big file
    results['times'] = times
    results['frame_increments'] = frame_increments
    results['cn_filters'] = cn_filters
    results['pnr_list'] = pnr_list
    pkl.dump( results, open( "ResultsDB_F_Frame_Files_1.p", "wb" ) )

    # Test the big original file

    video_for_calc = cm.load(r'O:\archive\projects\2023_intercontext\PICAST\data\1_preprocessed\20230405_m1018_wt_1110\miniscope_video\memmap__d1_341_d2_398_d3_1_order_C_frames_37564.mmap')
    gsig = 5
    for frame_increment in range(1,0,-1):
        st = time.time()

        try:
            cn_filter, pnr = cm.summary_images.correlation_pnr(video_for_calc[::frame_increment], gSig=gsig, swap_dim=False)
        except:
            cn_filter = [0]
            pnr = [0]

        et = time.time()
        elapsed_time = et - st
        times.append(elapsed_time)
        cn_filters.append(cn_filter)
        pnr_list.append(pnr)
        frame_increments.append(100000+frame_increment)

    results['times'] = times
    results['frame_increments'] = frame_increments
    results['cn_filters'] = cn_filters
    results['pnr_list'] = pnr_list
    pkl.dump( results, open( "ResultsDB_F_Frame_Files_2.p", "wb" ) )

    # os.system("shutdown /s /t 1")


    # pkl.load( open ("ResultsDB_F_Frame_Files.p", "rb")

    # cm.utils.visualization.inspect_correlation_pnr(cn_filter, pnr)
    # cm.utils.visualization.inspect_correlation_pnr(cn_filters[i], pnr_list[i])
    # seaborn.kdeplot(cn_filter.flatten())


#% Test powerpoint plot generation code




