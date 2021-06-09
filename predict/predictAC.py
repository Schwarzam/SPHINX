import pandas as pd
import numpy as np
#import param_SPLUS82 as param
import sys,os

sys.path.append("../interface")
import dataset, net_functions, network_array
import io_functions

#################################################################
io_functions.intro()
### READ TRAINING SET
params = eval(open("../params/param_AC.py", 'r').read())

TEFF_train = dataset.Dataset(path = params['segue_path'],
                               variable='AC',
                               params = params,
                               mode='TRAINING')


TEFF_train.format_names()
TEFF_train.SNR_threshold(32)
TEFF_train.faint_bright_limit()
TEFF_train.error_reject(training=True)
TEFF_train.specify_variable_bounds('TEFF', bounds = [params['TEFF_MIN'], params['TEFF_MAX']])
TEFF_train.build_colors()
TEFF_train.set_variable_bounds(run=True)
TEFF_train.get_input_stats(inputs='colors')



### generate the important stuff

TEFF_train.uniform_kde_sample()

TEFF_train.get_input_stats(inputs='colors')

### Initialize network

AC_NET = network_array.Network_Array(TEFF_train, target_variable = "AC",
                                        interp_frame = None,
                                        scale_frame  = None,
                                        params       = params,
                                        input_type   = "colors")

AC_NET.load_state('AC_NET_5500_7000_lower.pkl.gz')






AC_pred = dataset.Dataset(path = params['target_path'],
                               variable='AC',
                               params = params,
                               mode='TARGET')

AC_pred.format_names()
AC_pred.build_colors()
AC_pred.get_input_stats(inputs='colors')

AC_NET.predict(AC_pred)
