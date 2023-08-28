"""
Some code to play with the DB structure
Author: Bashar
"""
#%% imports
    import datastructure_tools as dt
    from miniscope_pipeline_DB import MiniscopeDataBaseAccess
    from miniscope_pipeline_DB.MiniscopeDataBaseAccess import MiniscopePipeline
    import caiman as cm

#%% Setup Code
    _DB = dt.DataBaseAccess.DataBaseAccess()
    # start the cluster
    try:
        cm.stop_server()  # stop it if it was running
    except():
        pass

    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', # if error try to change
                                                     n_processes=16,  # number of process to use, if you go out of memory try to reduce this one
                                                     single_thread=False)

    #minipipe = MiniscopePipeline(dview = None)         # Start new cluster
    minipipe = MiniscopePipeline(dview = dview, n_processes = n_processes)   # Start with parallel processing


#%%
    """
    Sh*t to remember:
    minipipe.MiniscopeRecording * minipipe.DB.Session
    (minipipe.DB.Session & 'animal_id = "m1018_wt"').delete()
    (minipipe.DB.Session & 'animal_id = "m1018_wt"')
    minipipe.filter_sessions()
    'miniscope' in minipipe.DB.ExperimentTemplate.TemplateBuild().fetch('building_block')
    """


#%% a primitive simulation to understand the workflow of the miniscope pipeline
    minipipe.DB.Session.fetch()                 # to see the content of the table
    minipipe.insert_crop_params('m1018_wt')     # set crop parameter for the animal (with GUI)
    minipipe.Cropping.populate(suppress_errors = True, processes = 16, display_progress= True)                         # to perform the cropping task
    minipipe.MotionCorrectionParamSet.insert_new_params(mc_method = 'caiman_mc',
                                                        mc_paramset_desc = 'Start parameters for optimization',
                                                        mc_params= mc_dict)                                            # parameter set for Motion correction
    minipipe.MotionCorrection.populate()
    minipipe.autofill_PNR_SNR_task()
    minipipe.PNRCorrImage.populate(suppress_errors =True, display_progress=True)
    minipipe.autofill_extraction()
    minipipe.ComponentExtraction.populate(suppress_errors = True)
    minipipe.autofill_evaluation(**kwargs)
    minipipe.ComponentEvaluation.populate(suppress_errors =True)
    minipipe.ComponentExtractionPipelineOutput.populate(suppress_errors=True)

