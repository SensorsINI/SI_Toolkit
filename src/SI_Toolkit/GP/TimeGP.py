import timeit

gp_path = './SI_Toolkit_ASF/Experiments/Experiment-Xi/Models'
gp_name = 'SGP_10'

def timing_script_init():
    import tensorflow as tf
    from SI_Toolkit.GP.Functions.save_and_load import load_model

    save_dir = gp_path + "/" + gp_name + "/"

    # load model
    print("Loading...")
    m_loaded = load_model(save_dir)
    print("Done!")

    num_rollouts = 2000
    horizon = 35

    s = tf.zeros(shape=[num_rollouts, 6], dtype=tf.float64)
    m_loaded.predict_f(s)

    return m_loaded, s


if __name__ == '__main__':

    initialization = '''\
from SI_Toolkit.GP.TimeGP import timing_script_init
m_loaded, s = timing_script_init()
    '''

    code = '''\
mn = m_loaded.predict_f(s)
    '''

    print(timeit.timeit(code, number=35, setup=initialization))