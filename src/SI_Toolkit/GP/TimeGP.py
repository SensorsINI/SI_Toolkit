import timeit

## TIMING PREDICTION WITH LOADED MODEL
initialization = '''
import tensorflow as tf
import numpy as np
from SI_Toolkit.GP.Models import load_model
from SI_Toolkit.Functions.General.load_parameters_for_training import args

a = args()
save_dir = a.path_to_models + "/SGP_10/"

# load model
print("Loading...")
m_loaded = load_model(save_dir)
print("Done!")

num_rollouts = 2000
horizon = 35

s = tf.zeros(shape=[num_rollouts, 6], dtype=tf.float64)
m_loaded.predict_f(s)
'''

code = '''\
mn = m_loaded.predict_f(s)
'''

print(timeit.timeit(code, number=35, setup=initialization))

# plot_test(model, data_val, closed_loop=True)  # plot posterior predictions with loaded trained model

