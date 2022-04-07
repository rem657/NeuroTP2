import numpy as np
from typing import NamedTuple, Callable, Union
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

from src.ifunc import IConst, IFunc, ISin, ISinSquare


class LIFOutput(NamedTuple):
	"""
	Output of LIF network.
	"""
	T: int
	dt: float
	t_space: np.ndarray
	t_space_ms: np.ndarray
	v: np.ndarray
	spikes: np.ndarray
	g_E: np.ndarray
	g_I: np.ndarray
	I: np.ndarray


class LIF:
	def __init__(
			self,
			n_exc: int = 50,
			n_inh: int = 50,
			E_exc: float = None,
			E_inh: float = None,
			p_EE: float = 0.1,
			p_EI: float = 0.1,
			p_IE: float = 0.1,
			p_II: float = 0.1,
			v_thresh: float = -55.0,
			v_reset: float = -75.0,
			g_L: float = 1.0,
			E_L: float = -75.0,
			tau_syn: float = 10.0,
	):
		self.n_exc = n_exc
		self.n_inh = n_inh
		self.exc_indexes = np.arange(n_exc)
		self.inh_indexes = np.arange(n_exc, n_exc + n_inh)
		self.N_indexes = np.arange(n_exc + n_inh)
		
		self.E_exc = v_thresh + 15 if E_exc is None else E_exc
		self.E_inh = v_reset - 15 if E_inh is None else E_inh
		self.E = np.zeros(n_exc + n_inh, dtype=np.float32)
		self.E[self.exc_indexes] = self.E_exc
		self.E[self.inh_indexes] = self.E_inh
		
		self.p_EE = p_EE
		self.p_EI = p_EI
		self.p_IE = p_IE
		self.p_II = p_II
		self.c = np.zeros((n_exc + n_inh, n_exc + n_inh), dtype=np.float32)
		self.set_p(p_EE, p_EI, p_IE, p_II)
		
		self.v_thresh = v_thresh
		self.v_reset = v_reset
		self.g_L = g_L
		self.E_L = E_L
		self.tau_syn = tau_syn

	def set_p(self, p_EE: float = None, p_EI: float = None, p_IE: float = None, p_II: float = None):
		if p_EE is not None:
			self.p_EE = p_EE
		if p_EI is not None:
			self.p_EI = p_EI
		if p_IE is not None:
			self.p_IE = p_IE
		if p_II is not None:
			self.p_II = p_II

		self.c = np.zeros((self.n_exc + self.n_inh, self.n_exc + self.n_inh), dtype=np.float32)
		self.c[np.ix_(self.exc_indexes, self.exc_indexes)] = (
				np.random.rand(self.n_exc, self.n_exc) < self.p_EE
		).astype(np.float32)
		self.c[np.ix_(self.exc_indexes, self.inh_indexes)] = (
				np.random.rand(self.n_exc, self.n_inh) < self.p_EI
		).astype(np.float32)
		self.c[np.ix_(self.inh_indexes, self.exc_indexes)] = (
				np.random.rand(self.n_inh, self.n_exc) < self.p_IE
		).astype(np.float32)
		self.c[np.ix_(self.inh_indexes, self.inh_indexes)] = (
				np.random.rand(self.n_inh, self.n_inh) < self.p_II
		).astype(np.float32)

		np.fill_diagonal(self.c, 0.0)
	
	def get_neuron_type(self, n_idx: int):
		"""
		Get the type of the neuron with the given index.
		:param n_idx: The index of the neuron.
		:return: The type of the neuron with the given index.
		"""
		if n_idx in self.exc_indexes:
			return 'exc'
		elif n_idx in self.inh_indexes:
			return 'inh'
		else:
			raise ValueError(f'Unknown neuron index: {n_idx}')

	def get_neuron_indexes(self, n_type: str):
		"""
		Get the indexes of the neurons of the given type.
		:param n_type: The type of the neurons. Must be 'exc' or 'inh'.
		:return: The indexes of the neurons of the given type.
		"""
		if n_type == 'exc':
			return self.exc_indexes
		elif n_type == 'inh':
			return self.inh_indexes
		else:
			raise ValueError(f'Unknown neuron type: {n_type}')

	def __call__(self, *args, **kwargs):
		return self.run(*args, **kwargs)
	
	def run(
			self,
			I_exc_func: IFunc = IConst(0.0),
			I_inh_func: IFunc = IConst(0.0),
			T_ms: int = 100,
			dt=1e-3,
			verbose=True
	):
		time_steps = int(T_ms / dt)
		t_space = np.arange(start=0, stop=int(T_ms / dt), step=1, dtype=int)
		t_space_ms = np.arange(start=0, stop=T_ms, step=dt, dtype=np.float32)
		
		v_trace = self.v_reset * np.ones((time_steps, *self.E.shape), dtype=np.float32)
		g_E_trace = np.zeros((time_steps, *self.E.shape), dtype=np.float32)
		g_I_trace = np.zeros((time_steps, *self.E.shape), dtype=np.float32)
		spikes_trace = np.zeros((time_steps, *self.E.shape), dtype=np.float32)
		
		I_in_trace = np.zeros((time_steps, *self.E.shape), dtype=np.float32)
		I_in_trace[:, self.exc_indexes] = np.expand_dims(I_exc_func(t_space_ms), axis=-1)
		I_in_trace[:, self.inh_indexes] = np.expand_dims(I_inh_func(t_space_ms), axis=-1)
		
		for t in tqdm.tqdm(t_space[:-1], disable=not verbose):
			I_syn = g_E_trace[t]*(self.E_exc - v_trace[t]) + g_I_trace[t] * (self.E_inh - v_trace[t])
			dvdt = self.g_L*(self.E_L - v_trace[t]) + I_syn + I_in_trace[t]
			v_trace[t+1] = v_trace[t] + dt * dvdt
			spikes_trace[t+1] = np.where(v_trace[t+1] >= self.v_thresh, 1.0, 0.0)
			v_trace[t + 1] = np.where(np.isclose(spikes_trace[t+1], 1.0), self.v_reset, v_trace[t+1])
			
			g_E_syn = np.dot(self.c[np.ix_(self.N_indexes, self.exc_indexes)], spikes_trace[t][self.exc_indexes])
			g_E_trace[t+1] = g_E_trace[t]*(1-dt/self.tau_syn) + g_E_syn
			
			g_I_syn = np.dot(self.c[np.ix_(self.N_indexes, self.inh_indexes)], spikes_trace[t][self.inh_indexes])
			g_I_trace[t+1] = g_I_trace[t] * (1 - dt / self.tau_syn) + g_I_syn

		return LIFOutput(
			T=T_ms,
			dt=dt,
			t_space=t_space,
			t_space_ms=t_space_ms,
			v=v_trace,
			spikes=spikes_trace,
			g_E=g_E_trace,
			g_I=g_I_trace,
			I=I_in_trace
		)

	@staticmethod
	def get_firing_rate(lif_output: LIFOutput, reduce: bool = False) -> Union[np.ndarray, float]:
		"""
		Compute and return the firing rate given the lif output.
		:param lif_output: The given lif output.
		:param reduce: If True return the mean of the firing rates.
		:return: The firing rate.
		"""
		firing_rate = np.mean(lif_output.spikes, axis=0)
		if reduce:
			firing_rate = np.mean(firing_rate)
		return firing_rate

	def show_c(self):
		ax = sns.heatmap(self.c)
		plt.show()


def question_2_exploration():
	from src.ifunc import IConst

	np.random.seed(42)
	lif = LIF(n_exc=50, n_inh=50, p_EE=0.5, p_EI=0.1, p_IE=0.1, p_II=0.1, tau_syn=0.1)
	lif.show_c()
	lif_output = lif.run(IConst(15.0), IConst(0.0), T_ms=100, dt=0.001)

	n_neurone = 4
	fr_mean = LIF.get_firing_rate(lif_output, reduce=True)
	neuron_indexes = np.argsort(np.abs(fr_mean - LIF.get_firing_rate(lif_output, reduce=False)))[:n_neurone]
	neuron_indexes = np.argsort(LIF.get_firing_rate(lif_output, reduce=False))[-n_neurone:]
	# neuron_indexes = [0, ]
	fig, axes = plt.subplots(4, len(neuron_indexes), figsize=(8, 5), sharex='all', sharey='row')
	if isinstance(axes, plt.Axes):
		axes = np.asarray([[axes]])
	if axes.ndim < 2:
		axes = np.expand_dims(axes, axis=-1)
	for i, neuron_idx in enumerate(neuron_indexes):
		axes[0][i].set_title(f"Neuron {lif.get_neuron_type(neuron_idx)} {neuron_idx}")
		axes[0][i].plot(lif_output.t_space_ms, lif_output.v[:, neuron_idx])
		axes[1][i].plot(lif_output.t_space_ms, lif_output.g_E[:, neuron_idx])
		axes[2][i].plot(lif_output.t_space_ms, lif_output.g_I[:, neuron_idx])
		axes[3][i].plot(lif_output.t_space_ms, lif_output.I[:, neuron_idx])

	axes[0][0].set_ylabel("V [mV]")
	axes[1][0].set_ylabel("g_E [?]")
	axes[2][0].set_ylabel("g_I [?]")
	axes[3][0].set_ylabel("I [?]")
	axes[-1][int(len(neuron_indexes) / 2)].set_xlabel("Time [ms]")
	plt.tight_layout(pad=2.0)
	plt.show()


def question_2a_worker(lif, p_EE, I_E_value):
	lif.set_p(p_EE=p_EE)
	lif_output = lif.run(I_exc_func=IConst(I_E_value), T_ms=400, dt=1e-2, verbose=False)
	return LIF.get_firing_rate(lif_output, reduce=True)


def question_2a():
	from src.ifunc import IConst
	from pythonbasictools.multiprocessing import apply_func_multiprocess

	sns.set_theme()
	np.random.seed(42)

	p_EE_space = np.linspace(0.0, 1.0, num=10)
	I_E_value_space = np.linspace(10.0, 30.0, num=10)

	pp, II = np.meshgrid(p_EE_space, I_E_value_space)

	# store all the combination of p_EE and I_E
	params_space = np.asarray((pp, II)).T.reshape(-1, 2)

	lif = LIF(n_exc=50, n_inh=50, p_EE=0.5, p_EI=0.5, p_IE=0.5, p_II=0.5)
	firing_rates_list = apply_func_multiprocess(
		func=question_2a_worker,
		iterable_of_args=[(lif, p_EE, I_E_value) for p_EE, I_E_value in params_space]
	)
	# firing_rates_list = [
	# 	question_2a_worker(lif, p_EE, I_E_value) for p_EE, I_E_value in params_space
	# ]

	ax = sns.heatmap(
		np.asarray(firing_rates_list).reshape((*p_EE_space.shape, *I_E_value_space.shape))[::-1, :],
		xticklabels=['{:,.2f}'.format(x) for x in p_EE_space],
		yticklabels=['{:,.2f}'.format(x) for x in I_E_value_space[::-1]],
		cbar_kws={'label': 'Mean spiking rate'}
	)
	ax.set_xlabel("p_EE")
	ax.set_ylabel("I_E")
	plt.show()


if __name__ == '__main__':
	question_2_exploration()
	question_2a()





