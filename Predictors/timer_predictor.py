import timeit
import sys, os
from tqdm import trange
import numpy as np

number = 5000  # FIXME: Why time per call get lower when number of calls grows? It seems you need 100-1000 to get minimal values
repeat = 5

def timer_predictor(initialisation):
    code = '''\
predictor.predict(initial_state, Q)
predictor.update_internal_state(initial_state, Q)'''

    class HiddenPrints:
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout

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