import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from typing import Callable, Any, List, Tuple, Union
import seaborn as sns

plot_layout = dict(
	plot_bgcolor='aliceblue',
	paper_bgcolor="white",
	xaxis=dict(
		showgrid=False,
		zeroline=False,
		title_font={'size': 28},
		tickfont=dict(
			size=30
		)
	),
	yaxis=dict(
		showgrid=False,
		zeroline=False,
		title_font={'size': 28},
		tickfont=dict(
			size=30
		)
	),
	legend=dict(
		font=dict(
			size=25
		)
	)
)
axes_3D = dict(
	showgrid=False,
	zeroline=False,
	title_font={'size': 30},
	tickfont=dict(
		size=20,
	),
	backgroundcolor='aliceblue',
	showspikes=False,
	spikesides=False
)

plot_layout_3D = dict(
	paper_bgcolor="white",
	scene=dict(
		xaxis=axes_3D,
		yaxis=axes_3D,
		zaxis=axes_3D,
	),
	legend=dict(
		font=dict(
			size=22
		)
	)
)

E_colorscale_name = 'gist_heat'  # 'inferno'#'rocket'#'OrRd'
I_colorscale_name = 'ocean'  # 'mako'#'winter_r'#'PuBuGn_r'

contours3D = dict(
	x=dict(
		show=False,
		highlight=False
	),
	y=dict(
		show=False,
		highlightcolor="black",
		highlightwidth=16,
		highlight=True
	),
	z=dict(
		show=False,
		highlight=False
	)
)


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
		A = A * np.array([1, -1])
		return I + np.dot(self.weights, A)

	def dAdt(self, S: np.ndarray, A: np.ndarray, I: np.ndarray):
		return self.gamma * self.F(self.INP(A, I)) * S - self.alpha * A

	def dSdt(self, A: np.ndarray, S: np.ndarray, R: np.ndarray, I: np.ndarray):
		return self.beta * R - self.gamma * self.F(self.INP(A, I)) * S

	def dRdt(self, A: np.ndarray, R: np.ndarray):
		return -self.beta * R + self.alpha * A

	def dXdt(self, t, X: np.ndarray, I: callable):
		len_group = int(len(X) / 3)
		A = X[: len_group]
		S = X[len_group: 2 * len_group]
		R = X[2 * len_group: 3 * len_group]
		dadt = self.dAdt(S, A, I(t))
		dsdt = self.dSdt(A, S, R, I(t))
		drdt = self.dRdt(A, R)
		# print(dadt)
		# print(dsdt)
		# print(drdt)
		return np.array([*dadt, *dsdt, *drdt])

	def compute_model(self, init_cond: np.ndarray, I: callable, **kwargs):
		return solve_ivp(self.dXdt, self.time, init_cond, method='LSODA', args=[I], **kwargs)

	def nullcline_A_E(self, A_e, I_e):
		gamma_alpha = self.gamma / self.alpha
		Ae_Se = (1 / A_e) - 1 - (self.alpha / self.beta)
		a = gamma_alpha * Ae_Se
		a[a == 0] = None
		ln = a - 1
		Wei = self.weights[0, 1]
		Wee = self.weights[0, 0]
		return (1 / Wei) * (np.log(ln) + I_e + A_e * Wee)

	def nullcline_A_I(self, A_i, I_i):
		gamma_alpha = self.gamma / self.alpha
		Ai_Si = (1 / A_i) - 1 - (self.alpha / self.beta)
		a = gamma_alpha * Ai_Si
		a[a == 0] = None
		ln = a - 1
		Wie = self.weights[1, 0]
		Wii = self.weights[1, 1]
		return (1 / Wie) * (A_i * Wii - np.log(ln) - I_i)


# gammaSe = self.gamma * (1 - A_e - (self.alpha / self.beta) * A_e)
# alphaAe = self.alpha * A_e
# alphaAe[alphaAe == 0] = None
# a1 = gammaSe / alphaAe
# ln = a1 - 1  # - np.exp(-I_e) - np.exp(-A_e*self.weights[0, 0])
# a2 = I_e + A_e * self.weights[0, 0]
# return   # (self.gamma/self.alpha) * self.F(self.INP(1-S+R, I)) * S


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


def _make_surfaces_WC(
		weights: np.ndarray,
		I_min: float,
		I_max: float,
		n_step_I: int,
		t_min: float = 0.0,
		t_max: float = 50.0,
		step_size_t: float = 0.5,
		alpha: float = 1,
		gamma: float = 0.2,
		beta: float = 0.2,
):
	model = WilsonCowanModel(t_min, t_max, gamma, alpha, beta, weights)
	A = [0.0, 0.0]
	S = [1.0, 1.0]
	R = [0.0, 0.0]
	init_cond = np.array([*A, *S, *R])
	currents = np.linspace(I_min, I_max, n_step_I)
	time = np.arange(t_min, t_max, step_size_t)
	data_A_E = pd.DataFrame(np.zeros((n_step_I, time.shape[0])), columns=time, index=currents)
	data_A_I = pd.DataFrame(np.zeros((n_step_I, time.shape[0])), columns=time, index=currents)
	data_S_E = pd.DataFrame(np.zeros((n_step_I, time.shape[0])), columns=time, index=currents)
	data_S_I = pd.DataFrame(np.zeros((n_step_I, time.shape[0])), columns=time, index=currents)
	data_R_E = pd.DataFrame(np.zeros((n_step_I, time.shape[0])), columns=time, index=currents)
	data_R_I = pd.DataFrame(np.zeros((n_step_I, time.shape[0])), columns=time, index=currents)
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
	return dataf


def display_surfaces_WC(
		weights: np.ndarray,
		I_min: float,
		I_max: float,
		n_step_I: int,
		t_min: float = 0.0,
		t_max: float = 50.0,
		step_size_t: float = 0.5,
		alpha: float = 1,
		gamma: float = 0.2,
		beta: float = 0.2,
		save: bool = False
):
	dataf = _make_surfaces_WC(weights, I_min, I_max, n_step_I, t_min, t_max, step_size_t, alpha, gamma, beta)
	figure = go.Figure()
	colorscale_E = [[i / (int(len(dataf) / 2) - 1), f'rgb{tuple(map(lambda c: int(255 * c), color_E))}'] for i, color_E
	                in enumerate(sns.color_palette(E_colorscale_name, int(len(dataf) / 2)))]
	colorscale_I = [[j / (int(len(dataf) / 2) - 1), f'rgb{tuple(map(lambda c: int(255 * c), color_I))}'] for j, color_I
	                in enumerate(sns.color_palette(I_colorscale_name, int(len(dataf) / 2)))]
	count_E, count_I = 0, 0
	for df in dataf:
		temp_data = dataf[df]
		t = temp_data.columns
		I = temp_data.index
		prop = temp_data.values
		colors = np.ones(prop.shape)
		is_excit = df.endswith('E')
		if is_excit:
			colors = colors * (count_E / (int(len(dataf) / 2) - 1))
			count_E += 1
			colorscale = colorscale_E
		# print(colorscale)
		else:
			colors = colors * (count_I / (int(len(dataf) / 2) - 1))
			count_I += 1
			colorscale = colorscale_I
		figure.add_trace(
			go.Surface(
				x=t,
				y=I,
				z=prop,
				showscale=False,
				name=df[-3:],
				opacity=0.7,
				colorscale=colorscale,
				surfacecolor=colors,
				showlegend=True,
				cmin=0,
				cmax=1,
				contours=contours3D
			)
		)
	figure.update_layout(
		scene=dict(
			xaxis=dict(
				title='Time [ms]',
				range=[t_min, t_max]
			),
			yaxis=dict(
				title='I_E [-]',
				range=[I_min, I_max]
			),
			zaxis=dict(
				title='Proportion [-]',
				range=[0, 1]
			)
		)
	)
	figure.update_layout(plot_layout_3D)
	if save:
		figure.write_html('figure1a.html')
	else:
		figure.show()


def _make_surfaces_wee_time(
		I_e: float,
		wee_min: float,
		wee_max: float,
		wee_step_size: int,
		t_min: float,
		t_max: float,
		step_size_t: float,
		gamma: float = 0.2,
		alpha: float = 1.0,
		beta: float = 0.2
):
	A = [0.0, 0.0]
	S = [1.0, 1.0]
	R = [0.0, 0.0]
	init_cond = np.array([*A, *S, *R])
	time = np.arange(t_min, t_max, step_size_t)
	weights = np.arange(wee_min, wee_max, wee_step_size)
	data_A_E = pd.DataFrame(np.zeros((weights.shape[0], time.shape[0])), columns=time, index=weights)
	data_A_I = pd.DataFrame(np.zeros((weights.shape[0], time.shape[0])), columns=time, index=weights)
	data_S_E = pd.DataFrame(np.zeros((weights.shape[0], time.shape[0])), columns=time, index=weights)
	data_S_I = pd.DataFrame(np.zeros((weights.shape[0], time.shape[0])), columns=time, index=weights)
	data_R_E = pd.DataFrame(np.zeros((weights.shape[0], time.shape[0])), columns=time, index=weights)
	data_R_I = pd.DataFrame(np.zeros((weights.shape[0], time.shape[0])), columns=time, index=weights)
	current_func = lambda t: np.array([I_e, 0])
	for weight in weights:
		weight_matrix = np.array(
			[
				[weight, 0],
				[0, 0]
			]
		)
		model = WilsonCowanModel(t_min, t_max, gamma, alpha, beta, weight_matrix)
		solution = model.compute_model(init_cond, current_func, t_eval=time)
		y = solution.y
		data_A_E.loc[weight, :] = y[0, :]
		data_A_I.loc[weight, :] = y[1, :]
		data_S_E.loc[weight, :] = y[2, :]
		data_S_I.loc[weight, :] = y[3, :]
		data_R_E.loc[weight, :] = y[4, :]
		data_R_I.loc[weight, :] = y[5, :]
	dataf = dict(
		data_A_E=data_A_E,
		data_A_I=data_A_I,
		data_S_E=data_S_E,
		data_S_I=data_S_I,
		data_R_E=data_R_E,
		data_R_I=data_R_I,
	)
	return dataf


def display_surface_wee_time(
		I_e: float,
		wee_min: float,
		wee_max: float,
		wee_step_size: int,
		t_min: float,
		t_max: float,
		step_size_t: float,
		gamma: float = 0.2,
		alpha: float = 1.0,
		beta: float = 0.2,
		save: bool = False
):
	dataf = _make_surfaces_wee_time(
		I_e,
		wee_min,
		wee_max,
		wee_step_size,
		t_min,
		t_max,
		step_size_t,
		gamma,
		alpha,
		beta
	)
	figure = go.Figure()
	colorscale_E = [[i / (int(len(dataf) / 2) - 1), f'rgb{tuple(map(lambda c: int(255 * c), color_E))}'] for i, color_E
	                in enumerate(sns.color_palette(E_colorscale_name, int(len(dataf) / 2)))]
	colorscale_I = [[j / (int(len(dataf) / 2) - 1), f'rgb{tuple(map(lambda c: int(255 * c), color_I))}'] for j, color_I
	                in enumerate(sns.color_palette(I_colorscale_name, int(len(dataf) / 2)))]
	count_E, count_I = 0, 0
	for df in dataf:
		temp_data = dataf[df]
		t = temp_data.columns
		I = temp_data.index
		prop = temp_data.values
		colors = np.ones(prop.shape)
		is_excit = df.endswith('E')
		if is_excit:
			colors = colors * (count_E / (int(len(dataf) / 2) - 1))
			count_E += 1
			colorscale = colorscale_E
		# print(colorscale)
		else:
			colors = colors * (count_I / (int(len(dataf) / 2) - 1))
			count_I += 1
			colorscale = colorscale_I
		figure.add_trace(
			go.Surface(
				x=t,
				y=I,
				z=prop,
				showscale=False,
				name=df[-3:],
				opacity=0.7,
				colorscale=colorscale,
				surfacecolor=colors,
				showlegend=True,
				contours=contours3D,
				cmin=0,
				cmax=1
			)
		)
	figure.update_layout(
		scene=dict(
			xaxis=dict(
				title='Time [ms]',
				range=[t_min, t_max],
				**axes_3D
			),
			yaxis=dict(
				title='W<sub>EE</sub> [-]',
				range=[wee_min, wee_max],
				**axes_3D
			),
			zaxis=dict(
				title='Proportion [-]',
				range=[0, 1],
				**axes_3D
			)
		)
	)
	figure.update_layout(plot_layout_3D)
	if save:
		figure.write_html('figure1a.html')
	else:
		figure.show()


def _make_surfaces(
		I_e,
		I_i,
		W_ee,
		W_ei,
		W_ie,
		W_ii,
		t_min,
		t_max,
		step_size_t,
		gamma,
		alpha,
		beta
):
	A = [0.0, 0.0]
	S = [1.0, 1.0]
	R = [0.0, 0.0]
	init_cond = np.array([*A, *S, *R])
	time = np.arange(t_min, t_max, step_size_t)
	if type(I_e) is np.ndarray:
		arr_param = np.copy(I_e)
		param_name = 'I_e'
	elif type(I_i) is np.ndarray:
		arr_param = np.copy(I_i)
		param_name = 'I_i'
	elif type(W_ee) is np.ndarray:
		arr_param = np.copy(W_ee)
		param_name = 'W_ee'
	elif type(W_ei) is np.ndarray:
		arr_param = np.copy(W_ei)
		param_name = 'W_ei'
	elif type(W_ie) is np.ndarray:
		arr_param = np.copy(W_ie)
		param_name = 'W_ie'
	else:
		arr_param = np.copy(W_ii)
		param_name = 'W_ii'
	tuple_shape = (arr_param.shape[0], time.shape[0])
	data_A_E = pd.DataFrame(np.zeros(tuple_shape), columns=time, index=arr_param)
	data_A_I = pd.DataFrame(np.zeros(tuple_shape), columns=time, index=arr_param)
	data_S_E = pd.DataFrame(np.zeros(tuple_shape), columns=time, index=arr_param)
	data_S_I = pd.DataFrame(np.zeros(tuple_shape), columns=time, index=arr_param)
	data_R_E = pd.DataFrame(np.zeros(tuple_shape), columns=time, index=arr_param)
	data_R_I = pd.DataFrame(np.zeros(tuple_shape), columns=time, index=arr_param)
	for param in arr_param:
		if param_name == 'I_e':
			I_e = param
		elif param_name == 'I_i':
			I_i = param
		elif param_name == 'W_ee':
			W_ee = param
		elif param_name == 'W_ei':
			W_ei = param
		elif param_name == 'W_ie':
			W_ie = param
		elif param_name == 'W_ii':
			W_ii = param
		current_func = lambda t: np.array([I_e, I_i])
		weight_matrix = np.array(
			[
				[W_ee, W_ei],
				[W_ie, W_ii]
			]
		)
		model = WilsonCowanModel(t_min, t_max, gamma, alpha, beta, weight_matrix)
		solution = model.compute_model(init_cond, current_func, t_eval=time)
		y = solution.y
		data_A_E.loc[param, :] = y[0, :]
		data_A_I.loc[param, :] = y[1, :]
		data_S_E.loc[param, :] = y[2, :]
		data_S_I.loc[param, :] = y[3, :]
		data_R_E.loc[param, :] = y[4, :]
		data_R_I.loc[param, :] = y[5, :]
	dataf = dict(
		data_A_E=data_A_E,
		data_A_I=data_A_I,
		data_S_E=data_S_E,
		data_S_I=data_S_I,
		data_R_E=data_R_E,
		data_R_I=data_R_I,
	)
	return dataf


def display_surface(
		I_e,
		I_i,
		W_ee,
		W_ei,
		W_ie,
		W_ii,
		t_min,
		t_max,
		step_size_t,
		gamma,
		alpha,
		beta
) -> go.Figure:
	figure = go.Figure()
	dataf = _make_surfaces(
		I_e,
		I_i,
		W_ee,
		W_ei,
		W_ie,
		W_ii,
		t_min,
		t_max,
		step_size_t,
		gamma,
		alpha,
		beta
	)
	axis_name = ''
	if type(I_e) is np.ndarray:
		axis_name = 'I<sub>e</sub> [-]'
	elif type(I_i) is np.ndarray:
		axis_name = 'I<sub>i</sub> [-]'
	elif type(W_ee) is np.ndarray:
		axis_name = 'W<sub>EE</sub> [-]'
	elif type(W_ei) is np.ndarray:
		axis_name = 'W<sub>EI</sub> [-]'
	elif type(W_ie) is np.ndarray:
		axis_name = 'W<sub>IE</sub> [-]'
	elif type(W_ii) is np.ndarray:
		axis_name = 'W<sub>II</sub> [-]'
	colorscale_E = [[i / (int(len(dataf) / 2) - 1), f'rgb{tuple(map(lambda c: int(255 * c), color_E))}'] for i, color_E
	                in enumerate(sns.color_palette(E_colorscale_name, int(len(dataf) / 2)))]
	colorscale_I = [[j / (int(len(dataf) / 2) - 1), f'rgb{tuple(map(lambda c: int(255 * c), color_I))}'] for j, color_I
	                in enumerate(sns.color_palette(I_colorscale_name, int(len(dataf) / 2)))]
	count_E, count_I = 0, 0
	for df in dataf:
		temp_data = dataf[df]
		t = temp_data.columns
		var = temp_data.index
		prop = temp_data.values
		colors = np.ones(prop.shape)
		is_excit = df.endswith('E')
		if is_excit:
			colors = colors * (count_E / (int(len(dataf) / 2) - 1))
			count_E += 1
			colorscale = colorscale_E
		# print(colorscale)
		else:
			colors = colors * (count_I / (int(len(dataf) / 2) - 1))
			count_I += 1
			colorscale = colorscale_I
		figure.add_trace(
			go.Surface(
				x=t,
				y=var,
				z=prop,
				showscale=False,
				name=df[-3:],
				opacity=0.7,
				colorscale=colorscale,
				surfacecolor=colors,
				showlegend=True,
				contours=contours3D,
				cmin=0,
				cmax=1
			)
		)
	figure.update_layout(
		scene=dict(
			xaxis=dict(
				title='Time [ms]',
				range=[t_min, t_max],
				**axes_3D
			),
			yaxis=dict(
				title=axis_name,
				**axes_3D
			),
			zaxis=dict(
				title='Proportion [-]',
				range=[0, 1],
				**axes_3D
			)
		)
	)
	figure.update_layout(plot_layout_3D)
	return figure


def question1a():
	weights = np.array(
		[
			[0, 0],
			[0, 0]
		]
	)
	I_I = 0
	I_E = np.linspace(-15, 15, num=100)
	gamma = 0.2
	alpha = 1
	beta = 0.2
	figure = display_surface(I_E, I_I, weights[0, 0], weights[0, 1], weights[1, 0], weights[1, 1], 0, 100, 0.5, gamma,
	                         alpha, beta)
	figure.write_html(f"figures/figure_1a.html")


def question1b():
	weights = np.array(
		[
			[0, 0],
			[0, 0]
		]
	)
	I_I = 0
	I_E = 0
	W_EE = np.linspace(0, 50, 25)
	gamma = 0.2
	alpha = 1
	beta = 0.2
	figure = display_surface(I_E, I_I, W_EE, weights[0, 1], weights[1, 0], weights[1, 1], 0, 100, 0.5, gamma,
	                         alpha, beta)
	figure.show()


# figure.write_html(f"figures/figure_1b.html")


def _phase_plan(model: WilsonCowanModel, time: np.ndarray, current_fun: Callable, initial_cond: np.ndarray):
	solution = model.compute_model(initial_cond, current_fun, t_eval=time)
	y = solution.y
	A_E = y[0, :]
	A_I = y[1, :]
	return time, A_E, A_I


def phase_plan_layout(fig: go.Figure):
	fig.update_xaxes(
		title='A<sub>E</sub>',
		range=[0, 1]
	)
	fig.update_yaxes(
		title='A<sub>I</sub>',
		range=[0, 1]
	)
	fig.update_layout(plot_layout)
	return fig


def phase_plan_var_init(weights: np.ndarray, current_func: Callable, animate=False,
                        colorscale: str = 'twilight_shifted', save: bool = False):
	arr_init_conds = []
	# Condition réaliste
	A_0 = [0.0, 0.0]
	S_0 = [1.0 - A_0[0], 1.0 - A_0[1]]
	R_0 = [0.0, 0.0]
	arr_init_conds.append(np.array([*A_0, *S_0, *R_0]))
	A_1 = [1.0, 1.0]
	S_1 = [1.0 - A_1[0], 1.0 - A_1[1]]
	R_1 = [0.0, 0.0]
	arr_init_conds.append(np.array([*A_1, *S_1, *R_1]))
	A_right_8 = [0.55, 0.8]
	S_right_8 = [1 - A_right_8[0], 1 - A_right_8[1]]
	R_right_8 = [0.0, 0.0]
	arr_init_conds.append(np.array([*A_right_8, *S_right_8, *R_right_8]))
	A_right_10 = [0.9, 0.0]
	S_right_10 = [1 - A_right_10[0], 1 - A_right_10[1]]
	R_right_10 = [0.0, 0.0]
	arr_init_conds.append(np.array([*A_right_10, *S_right_10, *R_right_10]))
	A_right_5 = [0.9, 0.2]
	S_right_5 = [1 - A_right_5[0], 1 - A_right_5[1]]
	R_right_5 = [0.0, 0.0]
	arr_init_conds.append(np.array([*A_right_5, *S_right_5, *R_right_5]))
	A_right_1 = [0.9, 0.4]
	S_right_1 = [1 - A_right_1[0], 1 - A_right_1[1]]
	R_right_1 = [0.0, 0.0]
	arr_init_conds.append(np.array([*A_right_1, *S_right_1, *R_right_1]))
	A_right_3 = [0.9, 0.6]
	S_right_3 = [1 - A_right_3[0], 1 - A_right_3[1]]
	R_right_3 = [0.0, 0.0]
	arr_init_conds.append(np.array([*A_right_3, *S_right_3, *R_right_3]))
	A_center = [0.07, 0.01]
	S_center = [1 - A_center[0], 1 - A_center[1]]
	R_center = [0.0, 0.0]
	arr_init_conds.append(np.array([*A_center, *S_center, *R_center]))
	A_asym = [0.1, 1]
	S_asym = [1 - A_asym[0], 1 - A_asym[1]]
	R_asym = [0.0, 0.0]
	arr_init_conds.append(np.array([*A_asym, *S_asym, *R_asym]))
	A_asym_1 = [0.3, 1.0]
	S_asym_1 = [1 - A_asym_1[0], 1 - A_asym_1[1]]
	R_asym_1 = [0.0, 0.0]
	arr_init_conds.append(np.array([*A_asym_1, *S_asym_1, *R_asym_1]))
	A_asym0250 = [0.25, 0.0]
	S_asym0250 = [1 - A_asym0250[0], 1 - A_asym0250[1]]
	R_asym0250 = [0.0, 0.0]
	arr_init_conds.append(np.array([*A_asym0250, *S_asym0250, *R_asym0250]))
	A_right_6 = [0.8, 0.3]
	S_right_6 = [1 - A_right_6[0], 1 - A_right_6[1]]
	R_right_6 = [0.0, 0.0]
	arr_init_conds.append(np.array([*A_right_6, *S_right_6, *R_right_6]))
	A_right_6 = [0.8, 0.6]
	S_right_6 = [1 - A_right_6[0], 1 - A_right_6[1]]
	R_right_6 = [0.0, 0.0]
	arr_init_conds.append(np.array([*A_right_6, *S_right_6, *R_right_6]))
	# Conditions sous population
	# A_right = [0.8, 0.6]
	# S_right = [1 - A_right[0], 0.2]
	# R_right = [0.0, 0.0]
	# arr_init_conds.append(np.array([*A_right, *S_right, *R_right]))
	# A_right = [0.8, 0.6]
	# S_right = [0.0, 1 - A_right[1]]
	# R_right = [0.0, 0.0]
	# arr_init_conds.append(np.array([*A_right, *S_right, *R_right]))
	# A_right = [0.8, 0.6]
	# S_right = [1.0, 1 - A_right[1]]
	# R_right = [0.0, 0.0]
	# arr_init_conds.append(np.array([*A_right, *S_right, *R_right]))
	# A_right = [0.8, 0.6]
	# S_right = [1 - A_right[0], 1.0]
	# R_right = [0.0, 0.0]
	# arr_init_conds.append(np.array([*A_right, *S_right, *R_right]))
	As = list(zip(*arr_init_conds))[0:2]
	arr_theta = np.arctan((np.array(As[1])) / (np.array(As[0])))
	arr_init_conds = np.array(arr_init_conds)
	sorted_init_cond_index = np.argsort(arr_theta)
	tmin = 0
	tmax = 200
	time = np.linspace(tmin, tmax, 8000)
	model = WilsonCowanModel(tmin, tmax, weights=weights)
	figure = _phase_plan_var(model, time, current_func, arr_init_conds[sorted_init_cond_index], animate, colorscale)
	figure = phase_plan_layout(figure)
	if save:
		figure.write_html('figures/phase_plan_init_cond.html')
	else:
		figure.show()


def _phase_plan_var(
		model: WilsonCowanModel,
		arr_time: np.array,
		current_func: Callable,
		init_conditions: list,
		animate: bool = True,
		colorscale: str = 'twilight_shifted',
		nullcline_color: List[str] = ['orange', 'crimson'],
		figure=None,
		**kwargs
) -> go.Figure:
	palette = sns.color_palette(colorscale, len(init_conditions))
	if figure is None:
		figure = go.Figure()
	nb_frames = 1000
	nb_time_per_frame = arr_time.shape[0] // nb_frames
	list_trace_frame = []
	for index, condition in enumerate(init_conditions):
		time, A_E, A_I = _phase_plan(model, arr_time, current_func, condition)
		sum_condition_E = np.sum(condition[[0, 2, 4]])
		sum_condition_I = np.sum(condition[[1, 3, 5]])
		is_E_less_two = sum_condition_E < 1
		is_E_more_two = sum_condition_E > 1
		is_I_less_two = sum_condition_I < 1
		is_I_more_two = sum_condition_I > 1
		name_ext = '_I_underpop' if is_I_less_two else '_I_overpop' if is_I_more_two else '_E_underpop' if is_E_less_two else '_E_overpop' if is_E_more_two else ''
		dict_data = dict(
			mode='lines',
			name=f'({condition[0]}, {condition[1]}){name_ext}',
			line=dict(
				width=3,
				color=f'rgb{tuple(map(lambda c: int(255 * c), palette[index]))}'
			),
			**kwargs
		)
		figure.add_trace(
			go.Scatter(
				x=A_E,
				y=A_I,
				**dict_data
			)
		)
		list_frame = []
		for i in range(0, time.shape[0], nb_time_per_frame):
			first_index = np.clip(i - 3 * nb_time_per_frame, 0, time.shape[0])
			list_frame.append(
				go.Scatter(
					x=A_E[first_index:i + nb_time_per_frame],
					y=A_I[first_index:i + nb_time_per_frame],
					**dict_data
				)
			)
		list_trace_frame.append(list_frame)
	if animate:
		list_frame_trace = list(zip(*list_trace_frame))
		frames = [go.Frame(data=figure.data)] + [go.Frame(data=frame) for frame in list_frame_trace]
		figure.update_layout(
			updatemenus=[
				dict(
					type="buttons",
					buttons=[
						dict(
							label="Play",
							method="animate",
							args=[
								None,
								{
									"frame": {"duration": 60, "redraw": False},
									"fromcurrent": True,
									"mode": "immediate",
									"transition": {"duration": 0, "easing": "linear"}
								}
							]
						),
						dict(
							label="Pause",
							method="animate",
							args=[
								[None],
								{
									"frame": {"duration": 0, "redraw": False},
									"mode": "immediate",
									"transition": {"duration": 0}
								}
							],
						),
						dict(
							label="Reset",
							method="animate",
							args=[
								None,
								{
									"frame": {"duration": 500000, "redraw": False},
									"fromcurrent": False,
									"mode": "immediate",
									"transition": {"duration": 0}
								}
							],
						),
					]
				),
			],
		)
		if figure.frames is None:
			figure.frames = frames
		else:
			figure.frames = list(figure.frames) + frames
	arr_nulcline_ae = np.append([np.linspace(1e-26, 0.01, 10000), np.linspace(0.01, 0.0907, 10000, endpoint=True)],
	                            np.linspace(0.0907, 0.090909091, 200000))
	arr_nulcline_ai = np.append([np.linspace(1e-26, 0.01, 10000), np.linspace(0.01, 0.09, 10000, endpoint=True)],
	                            np.linspace(0.09, 0.1, 50000, endpoint=True))
	nulcline_ae = model.nullcline_A_E(arr_nulcline_ae, current_func(0)[0])
	nulcline_ai = model.nullcline_A_I(arr_nulcline_ai, current_func(0)[1])
	figure.add_trace(
		go.Scatter(
			x=arr_nulcline_ae[~np.isnan(nulcline_ae)],
			y=nulcline_ae[~np.isnan(nulcline_ae)],
			mode='lines',
			name='nullcline A<sub>E</sub>',
			line_width=5,
			line_color=nullcline_color[0],
			**kwargs
		)
	)
	figure.add_trace(
		go.Scatter(
			x=nulcline_ai[~np.isnan(nulcline_ai)],
			y=arr_nulcline_ai[~np.isnan(nulcline_ai)],
			mode='lines',
			name='nullcline A<sub>I</sub>',
			line_width=5,
			line_color=nullcline_color[1],
			**kwargs
		)
	)
	return figure


def phase_plan_var_weight(
		W_ee: Union[float, np.ndarray],
		W_ei: Union[float, np.ndarray],
		W_ie: Union[float, np.ndarray],
		W_ii: Union[float, np.ndarray],
		current_func: Callable,
		animate: bool = False,
		colorscale: str = 'rocket',
		nullcline_colorscale: List[str] = ['Reds', 'Blues'],
		save: bool = False
) -> None:
	"""
	Plot the phase plane of the model with the given weight matrix.

	:param current_func:
	:param save:
	:param nullcline_colorscale:
	:param W_ee: weight of the excitatory-excitatory connections
	:param W_ei: weight of the excitatory-inhibitory connections
	:param W_ie: weight of the inhibitory-excitatory connections
	:param W_ii: weight of the inhibitory-inhibitory connections
	:param animate: if True, the figure will be animated
	:param colorscale: color scale to use
	"""
	tmin = 0
	tmax = 200
	time = np.linspace(tmin, tmax, 8000)
	# generate initial conditions
	arr_init_conds = []
	for init_A in [[0.25, 0.0], [1.0, 1.0], [0.8, 0.3], [0.3, 0.9]]:
		A = init_A
		S = [1 - A[0], 1 - A[1]]
		R = [0.0, 0.0]
		arr_init_conds.append(np.array([*A, *S, *R]))
	As = list(zip(*arr_init_conds))[0:2]
	arr_theta = np.arctan((np.array(As[1])) / (np.array(As[0])))
	arr_init_conds = np.array(arr_init_conds)
	sorted_init_cond_index = np.argsort(arr_theta)
	arr_init_conds = arr_init_conds[sorted_init_cond_index]
	nullcline_len_dict = {'A_E': 1, 'A_I': 1}
	weights = np.ones((4,)) * -1
	# find wich weight is an array
	weigth_array = None
	weigth_name = ''
	colorscale_weight_var = None
	for index, weight in enumerate([W_ee, W_ei, W_ie, W_ii]):
		if isinstance(weight, np.ndarray):
			if weigth_array is None:
				weigth_array = weight
				weigth_name = ['W_ee', 'W_ei', 'W_ie', 'W_ii'][index]
				nullcline_name = ['A_E', 'A_I'][index // 2]
				nullcline_len_dict[nullcline_name] = len(weight)
				colorscale_weight_var = list(map(lambda color: f"rgb{tuple(map(lambda c: int(255 * c), color))}",
				                                 sns.color_palette(nullcline_colorscale[index // 2],
				                                                   len(weigth_array))))
			else:
				raise ValueError('Only one array of weight can be given')
		else:
			weights[index] = weight
	weights = weights.reshape((2, 2))
	if weigth_array is None:
		raise ValueError('No weight array found')

	A_e_palette, A_i_palette = sns.color_palette(nullcline_colorscale[0], nullcline_len_dict['A_E']), \
	                           sns.color_palette(nullcline_colorscale[1], nullcline_len_dict['A_I'])
	# compute models
	colorscale_current = [[i / (len(colorscale_weight_var) - 1), colorscale_weight_var[i]] for i in
	                      range(len(colorscale_weight_var))]
	mask = weights == -1
	figure = go.Figure()
	for index, weight in enumerate(weigth_array):
		weights[mask] = weight
		nullcline_colors = [
			f"rgb{tuple(map(lambda c: int(255 * c), A_e_palette[index if nullcline_len_dict['A_E'] != 1 else 0]))}",
			f"rgb{tuple(map(lambda c: int(255 * c), A_i_palette[index if nullcline_len_dict['A_I'] != 1 else 0]))}"]
		model = WilsonCowanModel(tmin, tmax, weights=weights)
		figure = _phase_plan_var(
			model,
			time,
			current_func,
			arr_init_conds[sorted_init_cond_index],
			animate,
			colorscale,
			nullcline_colors,
			figure,
			legendgroup=f'{weight}',
			legendgrouptitle_text=f"{weigth_name} = {weight}",
		)
	figure.add_trace(
		go.Scatter(
			x=[0],
			y=[0],
			mode='markers',
			opacity=0.0,
			marker=dict(
				colorscale=colorscale_current,
				showscale=True,
				cmin=weigth_array[0],
				cmax=weigth_array[-1],
				colorbar=dict(
					thickness=55,
					ticks="outside",
					tickvals=[i for i in weigth_array],
					title=dict(
						text=f"{weigth_name}",
						font_size=32,
					),
					tickfont_size=28,
				)
			),
			showlegend=False,
		)
	)
	figure = phase_plan_layout(figure)
	figure.update_layout(legend=dict(
		groupclick="toggleitem",
		yanchor="top",
		y=0.99,
		xanchor="left",
		x=1.13
	)
	)
	if save:
		figure.write_html(f"figures/phase_plan_var_weight_{weigth_name}.html")
	figure.show()


def phase_plan_var_I(
		I_e: Union[float, np.ndarray],
		I_i: Union[float, np.ndarray],
		weights: np.ndarray,
		animate: bool = False,
		colorscale: str = 'rocket',
		nullcline_colorscale: List[str] = ['Reds', 'Blues'],
		save: bool = False
) -> None:
	"""
	Plot the phase plane of the model with the given input currents.

	:param nullcline_colorscale:
	:param save:
	:param I_e: current of the excitatory population
	:param I_i: current of the inhibitory population
	:param weights: weight matrix
	:param animate: if True, the figure will be animated
	:param colorscale: color scale to use
	"""
	tmin = 0
	tmax = 200
	time = np.linspace(tmin, tmax, 8000)
	# generate initial conditions
	arr_init_conds = []
	for init_A in [[0.25, 0.0], [1.0, 1.0], [0.8, 0.3], [0.3, 0.9]]:
		A = init_A
		S = [1 - A[0], 1 - A[1]]
		R = [0.0, 0.0]
		arr_init_conds.append(np.array([*A, *S, *R]))
	As = list(zip(*arr_init_conds))[0:2]
	arr_theta = np.arctan((np.array(As[1])) / (np.array(As[0])))
	arr_init_conds = np.array(arr_init_conds)
	sorted_init_cond_index = np.argsort(arr_theta)
	arr_init_conds = arr_init_conds[sorted_init_cond_index]

	arr_current = None
	current_name = ''
	current_len_dict = {'I_e': 1, 'I_i': 1}
	current_func_array = np.full((2,), np.nan)
	colorscale_current_var: list = None
	# find which current is an array
	for index, current in enumerate([I_e, I_i]):
		if isinstance(current, np.ndarray):
			if arr_current is not None:
				raise ValueError('Only one array of current can be given')
			else:
				arr_current = current
				current_name = ['I_e', 'I_i'][index]
				current_len_dict[current_name] = len(current)
				colorscale_current_var = list(map(lambda color: f"rgb{tuple(map(lambda c: int(255 * c), color))}",
				                                  sns.color_palette(nullcline_colorscale[index], len(current))))
		else:
			current_func_array[index] = current
	if arr_current is None:
		raise ValueError('No current array found')
	I_e_palette, I_i_palette = sns.color_palette(nullcline_colorscale[0], current_len_dict['I_e']), \
	                           sns.color_palette(nullcline_colorscale[1], current_len_dict['I_i'])
	figure = go.Figure()
	# compute models
	mask = np.isnan(current_func_array)
	colorscale_current = [[i / (len(colorscale_current_var) - 1), colorscale_current_var[i]] for i in
	                      range(len(colorscale_current_var))]
	for index, current in enumerate(arr_current):
		nullcline_colors = [
			f"rgb{tuple(map(lambda c: int(255 * c), I_e_palette[index if current_len_dict['I_e'] != 1 else 0]))}",
			f"rgb{tuple(map(lambda c: int(255 * c), I_i_palette[index if current_len_dict['I_i'] != 1 else 0]))}"]
		current_func_array[mask] = current
		current_func = lambda t: current_func_array
		model = WilsonCowanModel(tmin, tmax, weights=weights)
		figure = _phase_plan_var(
			model,
			time,
			current_func,
			arr_init_conds[sorted_init_cond_index],
			animate,
			colorscale,
			nullcline_colors,
			figure,
			legendgroup=f'{current}',
			legendgrouptitle_text=f"{current_name} = {current}",
		)
	figure.add_trace(
		go.Scatter(
			x=[0],
			y=[0],
			mode='markers',
			opacity=0.0,
			marker=dict(
				colorscale=colorscale_current,
				showscale=True,
				cmin=arr_current[0],
				cmax=arr_current[-1],
				colorbar=dict(
					thickness=55,
					ticks="outside",
					tickvals=[i for i in arr_current],
					title=dict(
						text=f"{current_name}",
						font_size=32,
					),
					tickfont_size=28,
				)
			),
			showlegend=False,
		)
	)
	figure = phase_plan_layout(figure)
	figure.update_layout(legend=dict(
		groupclick="toggleitem",
		yanchor="top",
		y=0.99,
		xanchor="left",
		x=1.13
	)
	)
	if save:
		figure.write_html(f"figure/phase_plan_var_{current_name}.html")
	else:
		figure.show()


def question1c():
	weights = np.array(
		[
			[100, 1000],
			[100, 0]
		]
	)
	I_I = -10
	I_E = 5
	I_func = lambda t: np.array([I_E, I_I])
	#variation of init_cond
	phase_plan_var_init(weights, I_func, animate=True, save=True)
	#variation of current I
	phase_plan_var_I(I_E, np.linspace(-14, -2, 4), weights, animate=False, save=True)
	# variation of current E
	phase_plan_var_I(np.linspace(-10, 80, 5), I_I, weights, animate=False, save=True)
	# variation of W_ee
	phase_plan_var_weight(np.linspace(0, 800, 5), weights[0, 1], weights[1, 0], weights[1, 1], I_func, animate=False, save=True)
	# variation of W_ii
	phase_plan_var_weight(weights[0, 0], weights[0, 1], weights[1, 0], np.linspace(0, 225, 4), I_func, animate=False, save=True)
	# variation of W_ei
	phase_plan_var_weight(weights[0, 0], np.geomspace(100, 1000, 4), weights[1, 0], weights[1, 1], I_func, animate=False, save=True)
	# variation of W_ie
	phase_plan_var_weight(weights[0, 0], weights[0, 1], np.geomspace(100, 1000, 4), weights[1, 1], I_func, animate=False, save=True)


if __name__ == '__main__':
	weights = np.array(
		[
			[100, 1000],
			[100, 0]
		]
	)
	alpha = 1.0
	gamma = 0.2
	beta = 0.2
	I_func = lambda t: np.array([5, -10])
	# display_model(0, 100, weights, I_func, alpha, gamma, beta).show()
	# display_surfaces_WC(weights, -15, 15, 200, step_size_t=0.2)  # question 1a
	# display_surface_wee_time(0.0, 0.0, 55, 2, 0.0, 100.0, 0.2, )
	# phase_plan_var_init(weights, I_func, animate=True)
	# phase_plan_var_weight(weights[0, 0], weights[0, 1], np.geomspace(50, 1000, 5), weights[1, 1], I_func, animate=False, save=False)
	# phase_plan_var_I(np.linspace(-10, 80, 5), -10, weights, animate=False, save=False)
