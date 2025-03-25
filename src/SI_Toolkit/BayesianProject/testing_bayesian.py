import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from load_dataset import load_dataset
import tf_keras
from SI_Toolkit.load_and_normalize import load_data, get_paths_to_datafiles

tfd = tfp.distributions
negloglik = lambda y, rv_y: -rv_y.log_prob(y)

get_files_from = './Driver/CartPoleSimulation/SI_Toolkit_ASF/Experiments/AGRU_L_m_pole_full/Recordings/Test/'
paths_to_recordings = get_paths_to_datafiles(get_files_from)
dfs = load_data(list_of_paths_to_datafiles=paths_to_recordings, verbose=False)

print(dfs[0].columns)

time = dfs[0].loc[:, 'time'].to_numpy()
x_tst = dfs[0].loc[:, ['angle_sin', 'angle_cos']].to_numpy()
y_tst = dfs[0].loc[:, ['angle']].to_numpy()

# Load the saved model
model = tf_keras.models.load_model('bayesian_regression_model', 
                                   custom_objects={
                                       'KerasLayer': tf_keras.layers.Layer,
                                        '<lambda>': negloglik
                                   })


#@title Figure 4: Both Aleatoric & Epistemic Uncertainty
plt.figure(figsize=[6, 1.5])  # inches
# plt.plot(x, y, 'b.', label='observed');

yhats = [model(x_tst) for _ in range(100)]
avgm = np.zeros_like(x_tst[..., 0])
for i, yhat in enumerate(yhats):
  m = np.squeeze(yhat.mean())
  s = np.squeeze(yhat.stddev())
  if i < 15:
    plt.plot(time, m, 'r', label='ensemble means' if i == 0 else None, linewidth=1.)
    plt.plot(time, m + 2 * s, 'g', linewidth=0.5, label='ensemble means + 2 ensemble stdev' if i == 0 else None);
    plt.plot(time, m - 2 * s, 'g', linewidth=0.5, label='ensemble means - 2 ensemble stdev' if i == 0 else None);
  avgm += m
plt.plot(time, avgm/len(yhats), 'orange', label='overall mean', linewidth=4)
plt.plot(time, y_tst, 'b', label='true values', linewidth=4)

# plt.ylim(-0.,17);
# plt.yticks(np.linspace(0, 15, 4)[1:]);
# plt.xticks(np.linspace(*x_range, num=9));

ax=plt.gca();
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['left'].set_smart_bounds(True)
#ax.spines['bottom'].set_smart_bounds(True)
plt.legend(loc='center left', fancybox=True, framealpha=0., bbox_to_anchor=(1.05, 0.5))

plt.savefig('/tmp/fig4.png', bbox_inches='tight', dpi=300)

plt.show()