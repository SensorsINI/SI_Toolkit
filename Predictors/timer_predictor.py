import timeit
import sys, os
from tqdm import trange
import numpy as np

def timer_predictor(initialisation_specific, number=50, repeat=10):

    initialisation_start = '''
from SI_Toolkit_ApplicationSpecificFiles.predictors_customization import CONTROL_INPUTS
import numpy as np
batch_size = 2000
horizon = 50
net_name = 'GRU-6IN-32H1-32H2-5OUT-0' # if applies
# initial_state = np.random.random(size=(batch_size, 6))
initial_state = np.random.random(size=(1, 6))
Q = np.float32(np.random.random(size=(batch_size, horizon, len(CONTROL_INPUTS))))

'''
    initialisation_end = '''

predictor.predict(initial_state, Q)
predictor.update_internal_state(Q)
predictor.predict(initial_state, Q)
'''

    initialisation = initialisation_start+initialisation_specific+initialisation_end

    code = '''\
predictor.predict(initial_state, Q)
predictor.update_internal_state(Q)'''

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
    for i in range(repeat):
        with HiddenPrints():
            measurements.append(timer.timeit(number=number)/float(number))

    print()
    print('min: {:.4f} s'.format(np.min(measurements)))
    print('max: {:.4f} s'.format(np.max(measurements)))
    print('mean: {:.4f} s'.format(np.mean(measurements)))
    print('std: {:.4f} s'.format(np.std(measurements)))
    print()
    print("Measurements:")
    for measurement in measurements:
        print('{:.4f} s'.format(measurement))



if __name__ == '__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Restrict printing messages from TF

    initialisation_ODE = '''
from SI_Toolkit.Predictors.predictor_ODE import predictor_ODE
predictor = predictor_ODE(horizon, 0.02, 10)
'''

    initialisation_ODE_tf = '''
from SI_Toolkit.Predictors.predictor_ODE_tf import predictor_ODE_tf
predictor = predictor_ODE_tf(horizon, 0.02, 10)
'''

    initialisation_autoregressive_tf = '''
from SI_Toolkit.Predictors.predictor_autoregressive_tf import predictor_autoregressive_tf
predictor = predictor_autoregressive_tf(horizon, batch_size=batch_size, net_name=net_name, update_before_predicting=False)
'''

    initialisation_autoregressive_tf_integrated_update = '''
from SI_Toolkit.Predictors.predictor_autoregressive_tf import predictor_autoregressive_tf
predictor = predictor_autoregressive_tf(horizon, batch_size=batch_size, net_name=net_name, update_before_predicting=True)
'''

# Predictor of Jerome, now in others
#     initialisation_autoregressive_tf_Jerome = '''
# from SI_Toolkit.Predictors.predictor_autoregressive_tf_Jerome import predictor_autoregressive_tf
# predictor = predictor_autoregressive_tf(horizon, batch_size=batch_size, net_name='GRU-6IN-32H1-32H2-5OUT-0')
# '''


    # numbers = [1, 2]
    numbers = [1, 10, 100, 1000, 10000]

    print('Timing the predictors:')
    print('*************************')
    print('')

    for number in numbers:
        print('')
        print('------------------------------------------------------------------------------------------------------')
        print('')
        print('Number of calls: {}'.format(int(number)))
        print('')
        print('predictor_ODE')
        timer_predictor(initialisation_ODE)

        print('')
        print('')
        print('predictor_ODE_tf')
        timer_predictor(initialisation_ODE_tf)


        print('')
        print('')
        print('predictor_autoregressive_tf')
        timer_predictor(initialisation_autoregressive_tf)


        print('')
        print('')
        print('predictor_autoregressive_tf_integrated_update')
        timer_predictor(initialisation_autoregressive_tf_integrated_update)


        # print('')
        # print('')
        # print('predictor_autoregressive_tf_Jerome')
        # timer_predictor(initialisation_autoregressive_tf_Jerome)


    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Restrict printing messages from TF
