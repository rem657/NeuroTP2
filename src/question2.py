import numpy as np
from typing import NamedTuple, Callable

from src.ifunc import IFunc


class LIFOutput(NamedTuple):
	"""
	Output of LIF network.
	"""
	T: int
	dt: float
	v: np.ndarray
	spikes: np.ndarray
	g: np.ndarray
	I: np.ndarray


class LIF:
	def __init__(
			self,
			n_exc: int = 50,
			n_inh: int = 50,
			E_exc: float = 1.0,
			E_inh: float = -1.0,
			connection_prob: float = 0.1,
			v_thresh: float = 1.0,
			v_reset: float = 0.0,
			g_L: float = 0.1,
			E_L: float = 0.1,
			tau_syn: float = 0.1,
	):
		self.n_exc = n_exc
		self.n_inh = n_inh
		self.exc_indexes = np.arange(n_exc)
		self.inh_indexes = np.arange(n_exc, n_exc + n_inh)
		
		self.E = np.zeros(n_exc+ n_inh, dtype=np.float32)
		self.E[self.exc_indexes] = E_exc
		self.E[self.inh_indexes] = E_inh
		
		self.c = (np.random.rand(n_exc+n_inh, n_exc+n_inh) < connection_prob).astype(np.float32)
		self.v_thresh = v_thresh
		self.v_reset = v_reset
		self.g_L = g_L
		self.E_L = E_L
		self.tau_syn = tau_syn
	
	def run(self, I_exc_func: IFunc, I_inh: IFunc, T: int, dt=0.1):
		time_steps = int(T/dt)
		
		v_trace = np.zeros((time_steps, *self.E.shape), dtype=np.float32)
		g_trace = np.zeros((time_steps, *self.E.shape), dtype=np.float32)
		spikes_trace = np.zeros((time_steps, *self.E.shape), dtype=np.float32)
		
		I_trace = np.zeros((time_steps, *self.E.shape), dtype=np.float32)
		I_trace[:, self.exc_indexes] = np.expand_dims(
			I_exc_func(np.arange(start=0, stop=T, step=dt, dtype=np.float32)),
			axis=-1
		)
		I_trace[:, self.inh_indexes] = np.expand_dims(
			I_inh(np.arange(start=0, stop=T, step=dt, dtype=np.float32)),
			axis=-1
		)
		
		for t in range(time_steps-1):
			dvdt = self.g_L*(self.E_L - v_trace[t]) + g_trace[t]*(self.E - v_trace[t]) + I_trace[t]
			v_trace[t+1] = v_trace[t] + dt * dvdt
			spikes_trace[t+1] = np.where(v_trace[t+1] >= self.v_thresh, 1.0, 0.0)
			v_trace[t+1][spikes_trace[t+1] > 0.0] = self.v_reset
			g_trace = g_trace*(1-dt/self.tau_syn) - np.dot(self.c, spikes_trace[t])

		return LIFOutput(T=T, dt=dt, v=v_trace, spikes=spikes_trace, g=g_trace, I=I_trace)


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	from src.ifunc import IConst
	
	lif = LIF(n_exc=50, n_inh=50, connection_prob=0.1)
	lif_output = lif.run(IConst(1.0), IConst(1.0), T=100, dt=0.1)
	
	plt.figure(figsize=(10, 5))
	plt.subplot(1, 2, 1)
	plt.plot(lif_output.v[:, 0])
	plt.subplot(1, 2, 2)
	plt.plot(lif_output.v[:, 50])
	plt.show()





