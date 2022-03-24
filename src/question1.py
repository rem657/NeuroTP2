import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from typing import Callable


class WilsonCowanModel:
	def __init__(
			self,
			t_init: float,
			t_end: float,
			gamma: float = 0.2,
			alpha: float = 1,
			beta: float = 0.2,
			weights: np.ndarray = np.zeros((2, 2), dtype=float)
	):
		self.weights = weights  # [[W_EE, W_EI],[W_IE, W_II]]
		self.gamma = gamma
		self.alpha = alpha
		self.beta = beta
		self.time = (t_init, t_end)

	@staticmethod
	def F(x):
		return 1 / (1 + np.exp(-x))

	def INP(self, A: np.ndarray, I: np.ndarray):
		A = A * np.array([0, -1])
		return I + np.dot(self.weights, A)

	def dAdt(self, S: np.ndarray, A: np.ndarray, I: np.ndarray):
		return self.gamma * self.F(self.INP(A, I)) * S - self.alpha * A

	def dSdt(self, A: np.ndarray, S: np.ndarray, R: np.ndarray, I: np.ndarray):
		return self.beta * R - self.gamma * self.F(self.INP(A, I)) * S

	def dRdt(self, A: np.ndarray, R: np.ndarray):
		return -self.beta * R + self.alpha * A

	def dXdt(self, t, X: np.ndarray, I: callable):
		len_group = int(len(X)/3)
		A = X[: len_group]
		S = X[len_group: 2*len_group]
		R = X[2*len_group: 3*len_group]
		dadt = self.dAdt(S, A, I(t))
		dsdt = self.dSdt(A, S, R, I(t))
		drdt = self.dRdt(A, R)
		# print(dadt)
		# print(dsdt)
		# print(drdt)
		return np.array([*dadt, *dsdt, *drdt])

	def compute_model(self, init_cond: np.ndarray, I: callable, **kwargs):
		return solve_ivp(self.dXdt, self.time, init_cond, method='LSODA', args=[I], **kwargs)


def display_model(
		t_init: float,
		t_end: float,
		weights: np.ndarray,
		I: Callable,
		alpha: float = 1,
		gamma: float = 0.2,
		beta: float = 0.2
):
	model = WilsonCowanModel(t_init, t_end, gamma, alpha, beta, weights)
	A = [0.0, 0.0]
	S = [1.0, 1.0]
	R = [0.0, 0.0]
	init_cond = np.array([*A, *S, *R])
	solution = model.compute_model(init_cond, I)
	t = solution.t
	y = solution.y
	A_E = y[0, :]
	A_I = y[1, :]
	S_E = y[2, :]
	S_I = y[3, :]
	R_E = y[4, :]
	R_I = y[5, :]
	model_param = dict(
		A_E=A_E,
		A_I=A_I,
		S_E=S_E,
		S_I=S_I,
		R_E=R_E,
		R_I=R_I
	)
	fig = go.Figure()
	for param in model_param:
		fig.add_trace(
			go.Scatter(
				x=t,
				y=model_param[param],
				name=param,
				mode='lines',

			)
		)
	return fig


def display_model_I(
		weights: np.ndarray,
		I_min: float,
		I_max: float,
		n_step_I: int,
		t_min: float = 0.0,
		t_max: float = 50.0,
		n_step_t: int = 0.5,
		alpha: float = 1,
		gamma: float = 0.2,
		beta: float = 0.2,
		save: bool = False
):
	model = WilsonCowanModel(t_min, t_max, gamma, alpha, beta, weights)
	A = [0.0, 0.0]
	S = [1.0, 1.0]
	R = [0.0, 0.0]
	init_cond = np.array([*A, *S, *R])
	currents = np.linspace(I_min, I_max, n_step_I)
	time = np.linspace(t_min, t_max, n_step_t)
	data_A_E = pd.DataFrame(np.zeros((n_step_I, n_step_t)), columns=time, index=currents)
	data_A_I = pd.DataFrame(np.zeros((n_step_I, n_step_t)), columns=time, index=currents)
	data_S_E = pd.DataFrame(np.zeros((n_step_I, n_step_t)), columns=time, index=currents)
	data_S_I = pd.DataFrame(np.zeros((n_step_I, n_step_t)), columns=time, index=currents)
	data_R_E = pd.DataFrame(np.zeros((n_step_I, n_step_t)), columns=time, index=currents)
	data_R_I = pd.DataFrame(np.zeros((n_step_I, n_step_t)), columns=time, index=currents)
	for current in currents:
		current_func = lambda t: np.array([current, 0])
		solution = model.compute_model(init_cond, current_func, t_eval=time)
		y = solution.y
		data_A_E.loc[current, :] = y[0, :]
		data_A_I.loc[current, :] = y[1, :]
		data_S_E.loc[current, :] = y[2, :]
		data_S_I.loc[current, :] = y[3, :]
		data_R_E.loc[current, :] = y[4, :]
		data_R_I.loc[current, :] = y[5, :]
	dataf = dict(
		data_A_E=data_A_E,
		data_A_I=data_A_I,
		data_S_E=data_S_E,
		data_S_I=data_S_I,
		data_R_E=data_R_E,
		data_R_I=data_R_I,
	)
	figure = go.Figure()
	for df in dataf:
		temp_data = dataf[df]
		t = temp_data.columns
		I = temp_data.index
		prop = temp_data.values
		figure.add_trace(
			go.Surface(x=t, y=I, z=prop)
		)
		#todo colormap
	if save:
		figure.write_html('figure1a.html')
	else:
		figure.show()


if __name__ == '__main__':
	weights = np.array(
		[
			[0, 0],
			[0, 0]
		]
	)
	alpha = 1.0
	gamma = 0.2
	beta = 0.2
	I_func = lambda t: np.array([10, 0])
	# display_model(0, 100, weights, I_func, alpha, gamma, beta).show()
	display_model_I(weights, -15, 15, 200)