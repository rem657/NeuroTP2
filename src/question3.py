import numpy as np
from question2 import LIF, LIFOutput, IConst, plt, raster_plot


def lif_output_spike_rate(lif_output, n_E: int, n_I: int, time_window_size: int = 10):
	"""
	Returns the spike rate of a LIFOutput object
	"""
	spikes = lif_output.spikes
	time_space = lif_output.t_space
	time_space_ms = lif_output.t_space_ms
	step_window_size = int(time_window_size / lif_output.dt)
	time = []
	spike_rate_E = np.sum(spikes[:, np.arange(n_E)], axis=-1) / n_E
	spike_rate_I = np.sum(spikes[:, np.arange(n_E, n_I+n_E)], axis=-1) / n_I
	batch_time = time_space.reshape(-1, step_window_size)
	mean_spike_rate_E = np.zeros(batch_time.shape[0])
	mean_spike_rate_I = np.zeros(batch_time.shape[0])
	for i, indexes in enumerate(batch_time):
		mean_spike_rate_E[i] = np.mean(spike_rate_E[indexes])
		mean_spike_rate_I[i] = np.mean(spike_rate_I[indexes])
		time.append(time_space_ms[indexes[-1]])
	return time, mean_spike_rate_E, mean_spike_rate_I


def plot_raster_spike_rate(lif_output, n_E: int, n_I: int):
	"""
	Plots the raster plot and the spike rate of a LIFOutput object
	"""
	time, spike_rate_E, spike_rate_I = lif_output_spike_rate(lif_output, n_E, n_I, time_window_size=20)
	# time = lif_output.t_space_ms
	fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex='all')
	raster_plot(lif_output, ax=ax[0])
	ax[1].plot(time, spike_rate_I, 'b')
	ax[1].plot(time, spike_rate_E, 'r')
	ax[1].set_title('Spike rate of inhibitory and exhibitor neurons')
	ax[1].set_xlabel('Time (ms)')
	ax[1].set_ylabel('Spike rate (Hz)')
	fig.tight_layout()
	plt.show()


def raster_proportions(
		n_exc: int,
		n_inh: int,
		p_EE: float,
		p_EI: float,
		p_IE: float,
		p_II: float,
		I_E: float = 2.019,
		I_I: float = 0.0,
		T_ms: int = 400,
		dt: float = 0.01,
		save = False,
):
	"""
	Runs the question 3 code
	"""
	lif = LIF(n_exc=n_exc, n_inh=n_inh, p_EE=p_EE, p_EI=p_EI, p_IE=p_IE, p_II=p_II)
	lif_output = lif.run(I_exc_func=IConst(I_E), I_inh_func=IConst(I_I), T_ms=T_ms, dt=dt, verbose=False)
	plot_raster_spike_rate(lif_output, n_E=n_exc, n_I=n_inh)
	if save:
		plt.savefig(f'{p_EE=}_{p_EI=}_{p_IE=}_{p_II=}__{I_E=}_{I_I=}.png')
	plt.plot()


def question3b():
	#figure 1
	raster_proportions(50, 50, 0.8, 0.7, 0.6, 0.25, I_E=18, I_I=-20, T_ms=300, dt=0.025)
	# figure 2
	raster_proportions(50, 50, 0.8, 0.7, 0.6, 1, I_E=18, I_I=-20, T_ms=300, dt=0.025)

if __name__ == '__main__':
	raster_proportions(50, 50, 0.8, 0.7, 0.6, 1, I_E=18, I_I=-20, T_ms=300, dt=0.025)
