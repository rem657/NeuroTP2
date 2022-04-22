import numpy as np
from question2 import LIF, LIFOutput, IConst, plt, raster_plot


def lif_output_spike_rate(lif_output, n_E: int, n_I: int):
	"""
	Returns the spike rate of a LIFOutput object
	"""
	spikes = lif_output.spikes
	time_space = lif_output.t_space
	spike_rate_E = np.zeros(time_space.shape)
	spike_rate_I = np.zeros(time_space.shape)
	for i in time_space:
		spike_rate_E[i] = np.sum(spikes[i, np.arange(n_E)])/n_E
		spike_rate_I[i] = np.sum(spikes[i, np.arange(n_E, n_I+n_E)]) / n_I
	return spike_rate_E, spike_rate_I


def plot_raster_spike_rate(lif_output, n_E: int, n_I: int):
	"""
	Plots the raster plot and the spike rate of a LIFOutput object
	"""
	spike_rate_E, spike_rate_I = lif_output_spike_rate(lif_output, n_E, n_I)
	time = lif_output.t_space_ms
	fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
	raster_plot(lif_output, ax=ax[0])
	ax[1].plot(time, spike_rate_I, 'b')
	ax[1].plot(time, spike_rate_E, 'r')
	ax[1].set_title('Spike rate of inhibitory and exhibitor neurons')
	ax[1].set_xlabel('Time (ms)')
	ax[1].set_ylabel('Spike rate (Hz)')
	fig.tight_layout()
	plt.show()


def question3(
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
):
	"""
	Runs the question 3 code
	"""
	lif = LIF(n_exc=n_exc, n_inh=n_inh, p_EE=p_EE, p_EI=p_EI, p_IE=p_IE, p_II=p_II)
	lif_output = lif.run(I_exc_func=IConst(I_E), I_inh_func=IConst(I_I), T_ms=T_ms, dt=dt, verbose=False)
	plot_raster_spike_rate(lif_output, n_E=n_exc, n_I=n_inh)
	plt.plot()


if __name__ == '__main__':
	question3(50, 50, 0.2, 0.3, 0.6, 0.8, I_E=10, I_I=-10, T_ms=200, dt=0.025)
