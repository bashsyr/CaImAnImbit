    import matplotlib.pyplot as plt
    import numpy as np
    #different matplotlib backend for better plots
    import matplotlib
    matplotlib.rcParams['backend'] = 'Qt5Agg'  # works for win 11

    # Hotfix for win 11 plot issue
    #
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'qt')
    import pandas as pd
    import seaborn as sns




    Path = r'O:\archive\projects\2023_students\Result_files\grid_search\megan.pkl'
    df = pd.read_pickle(Path)    # to load the dataframe

