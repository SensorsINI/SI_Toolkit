import timeit
import sys, os
from tqdm import trange
import numpy as np

def timer_predictor(initialisation_specific, number=50, repeat=10):

    initialisation_start = '''
from SI_Toolkit_ASF.predictors_customization import CONTROL_INPUTS
import numpy as np
batch_size = 2000
horizon = 50
path_to_model = './SI_Toolkit_ASF/Experiments/CPS-17-02-2023-UpDown-Model/Models/'
net_name = 'Dense-6IN-32H1-32H2-5OUT-0' # if applies
GP_name = 'SGP_10'
initial_state = np.float32(np.random.random(size=(batch_size, 6)))
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

    initialisation_ODE_v0 = '''
from SI_Toolkit.Predictors.predictor_ODE_v0 import predictor_ODE_v0
predictor = predictor_ODE_v0(horizon, 0.02, 10)
'''

    initialisation_ODE = '''
from SI_Toolkit.Predictors.predictor_ODE import predictor_ODE
predictor = predictor_ODE(horizon=horizon, dt=0.02, intermediate_steps=10)
'''

    initialisation_autoregressive_neural = '''
from SI_Toolkit.Predictors.predictor_autoregressive_neural import predictor_autoregressive_neural
predictor = predictor_autoregressive_neural(horizon=horizon, batch_size=batch_size, model_name=net_name, path_to_model=path_to_model, update_before_predicting=False)
'''

    initialisation_autoregressive_neural_integrated_update = '''
from SI_Toolkit.Predictors.predictor_autoregressive_neural import predictor_autoregressive_neural
predictor = predictor_autoregressive_neural(horizon=horizon, batch_size=batch_size, model_name=net_name, path_to_model=path_to_model, update_before_predicting=True)
'''

    initialisation_autoregressive_GP = '''
from SI_Toolkit.Predictors.predictor_autoregressive_GP import predictor_autoregressive_GP
predictor = predictor_autoregressive_GP(horizon=horizon, batch_size=batch_size, model_name=GP_name, path_to_model=path_to_model, update_before_predicting=False)
    '''

# Predictor of Jerome, now in others
#     initialisation_autoregressive_tf_Jerome = '''
# from SI_Toolkit.Predictors.predictor_autoregressive_tf_Jerome import predictor_autoregressive_neural
# predictor = predictor_autoregressive_neural(horizon, batch_size=batch_size, net_name='GRU-6IN-32H1-32H2-5OUT-0')
# '''


    numbers = [10]
    # numbers = [1, 10, 100, 1000, 10000]

    print('Timing the predictors:')
    print('*************************')
    print('')

    for number in numbers:
        print('')
        print('------------------------------------------------------------------------------------------------------')
        print('')
        print('Number of calls: {}'.format(int(number)))


        print('')
        print('')
        print('predictor_ODE_v0')
        timer_predictor(initialisation_ODE_v0, number=number)


        print('')
        print('')
        print('predictor_ODE')
        timer_predictor(initialisation_ODE, number=number)


        print('')
        print('')
        print('predictor_autoregressive_neural')
        timer_predictor(initialisation_autoregressive_neural, number=number)


        print('')
        print('')
        print('predictor_autoregressive_tf_integrated_update')
        timer_predictor(initialisation_autoregressive_neural_integrated_update, number=number)


        print('')
        print('')
        print('predictor_autoregressive_GP')
        timer_predictor(initialisation_autoregressive_GP, number=number)

        # print('')
        # print('')
        # print('predictor_autoregressive_tf_Jerome')
        # timer_predictor(initialisation_autoregressive_tf_Jerome)


    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Restrict printing messages from TF
