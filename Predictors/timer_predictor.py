import timeit
import sys, os
from tqdm import trange
import numpy as np

# FIXME: Why time per call get lower when number of calls grows? It seems you need 100-1000 to get minimal values
def timer_predictor(initialisation_specific, number=50, repeat=2):

    initialisation_start = '''
from SI_Toolkit_ApplicationSpecificFiles.predictors_customization import CONTROL_INPUTS
import numpy as np
batch_size = 2000
horizon = 50
net_name = 'GRU-6IN-32H1-32H2-5OUT-0' # if applies
initial_state = np.random.random(size=(batch_size, 6))
# initial_state = np.random.random(size=(1, 6))
Q = np.float32(np.random.random(size=(batch_size, horizon, len(CONTROL_INPUTS))))

'''
    initialisation_end = '''

predictor.predict(initial_state, Q)
predictor.update_internal_state(initial_state, Q)
predictor.predict(initial_state, Q)
'''

    initialisation = initialisation_start+initialisation_specific+initialisation_end

    code = '''\
predictor.predict(initial_state, Q)
predictor.update_internal_state(initial_state, Q)'''

    class HiddenPrints:
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Restrict printing messages from TF

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Restrict printing messages from TF

    timer = timeit.Timer(code, setup=initialisation)
    measurements = []
    for i in trange(repeat):
        with HiddenPrints():
            measurements.append(timer.timeit(number=number)/float(number))

    print()
    print('min: {:.4f} s'.format(np.min(measurements)))
    print('max: {:.4f} s'.format(np.max(measurements)))
    print('mean: {:.4f} s'.format(np.mean(measurements)))
    print('std: {:.4f} s'.format(np.std(measurements)))
    print()
    for measurement in measurements:
        print('{:.4f} s'.format(measurement))

