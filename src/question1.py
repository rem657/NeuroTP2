import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from typing import Callable
import seaborn as sns

plot_layout = dict(
	plot_bgcolor='aliceblue',
	paper_bgcolor="white",
	xaxis=dict(
		showgrid=False,
		zeroline=False,
		title_font={'size': 20},
		tickfont=dict(
			size=20
		)
	),
	yaxis=dict(
		showgrid=False,
		zeroline=False,
		title_font={'size': 20},
		tickfont=dict(
			size=20
		)
	),
	legend=dict(
		font=dict(
			size=19
		)
	)
)
axes_3D = dict(
	showgrid=False,
	zeroline=False,
	title_font={'size': 20},
	tickfont=dict(
		size=18,
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
			size=19
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
		gamma_alpha = self.gamma/self.alpha
		Ae_Se = (1/A_e) - 1 - (self.alpha / self.beta)
		a = gamma_alpha * Ae_Se
		a[a==0] = None
		ln = a - 1
		Wei = self.weights[0, 1]
		Wee = self.weights[0, 0]
		return (1 / Wei) * (np.log(ln) + I_e + A_e*Wee)

	def nullcline_A_I(self, A_i, I_i):
		gamma_alpha = self.gamma/self.alpha
		Ai_Si = (1/A_i) - 1 - (self.alpha / self.beta)
		a = gamma_alpha * Ai_Si
		a[a==0] = None
		ln = a - 1
		Wie = self.weights[1, 0]
		Wii = self.weights[1, 1]
		return (1 / Wie) * (A_i*Wii - np.log(ln) - I_i)
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


def _phase_plan(model: WilsonCowanModel, time: np.ndarray, current_fun: Callable, initial_cond: np.ndarray):
	solution = model.compute_model(initial_cond, current_fun, t_eval=time)
	y = solution.y
	A_E = y[0, :]
	A_I = y[1, :]
	return time, A_E, A_I


def phase_plan_layout(fig: go.Figure):
	fig.update_xaxes(
		title='A<sub>E</sub>'
	)
	fig.update_yaxes(
		title='A<sub>I</sub>'
	)
	fig.update_layout(plot_layout)
	return fig


def phase_plan_var_init(weights: np.ndarray, current_func: Callable):
	# weights = np.array(
	# 	[
	# 		[100, 1000],
	# 		[100, 0]
	# 	]
	# )
	A_0 = [0.0, 0.0]
	S_0 = [1.0, 1.0]
	R_0 = [0.0, 0.0]
	init_cond_0 = np.array([*A_0, *S_0, *R_0])
	A_1 = [1.0, 1.0]
	S_1 = [0.0, 0.0]
	R_1 = [0.0, 0.0]
	init_cond_1 = np.array([*A_1, *S_1, *R_1])
	A_right = [0.7, 0.01]
	S_right = [1 - A_right[0], 1 - A_right[1]]
	R_right = [0.0, 0.0]
	init_cond_right = np.array([*A_right, *S_right, *R_right])
	A_right_2 = [0.7, 0.2]
	S_right_2 = [1 - A_right_2[0], 1 - A_right_2[0]]
	R_right_2 = [0.0, 0.0]
	init_cond_right_2 = np.array([*A_right_2, *S_right_2, *R_right_2])
	A_right_3 = [0.9, 0.3]
	S_right_3 = [1 - A_right_3[0], 1 - A_right_3[0]]
	R_right_3 = [0.0, 0.0]
	init_cond_right_3 = np.array([*A_right_3, *S_right_3, *R_right_3])
	A_right_3 = [0.9, 0.3]
	S_right_3 = [1 - A_right_3[0], 1 - A_right_3[0]]
	R_right_3 = [0.0, 0.0]
	init_cond_right_3 = np.array([*A_right_3, *S_right_3, *R_right_3])
	A_center = [0.07, 0.01]
	S_center = [1 - A_center[0], 1 - A_center[1]]
	R_center = [0.0, 0.0]
	init_cond_center = np.array([*A_center, *S_center, *R_center])
	A_asym = [0.1, 1]
	S_asym = [1 - A_asym[0], 1 - A_asym[1]]
	R_asym = [0.0, 0.0]
	init_cond_asym = np.array([*A_asym, *S_asym, *R_asym])
	A_asym_1 = [0.4, 0.8]
	S_asym_1 = [1 - A_asym_1[0], 1 - A_asym_1[1]]
	R_asym_1 = [0.0, 0.0]
	init_cond_asym_1 = np.array([*A_asym_1, *S_asym_1, *R_asym_1])

	x, y = np.mgrid[0:1:5j, 0:1:5j]
	x, y = x.flatten(), y.flatten()
	arr_init_conds = [np.array([x[i], y[i], 1 - x[i], 1 - y[i], 0.0, 0.0]) for i in range(x.shape[0])] + []
	# arr_init_conds = [init_cond_0, init_cond_1, init_cond_right, init_cond_center, init_cond_asym, init_cond_asym_1, init_cond_right_2, init_cond_right_3]
	figure = go.Figure()
	tmin = 0
	tmax = 200
	time = np.linspace(tmin, tmax, 8000)
	model = WilsonCowanModel(tmin, tmax, weights=weights)
	for condition in arr_init_conds:
		time, A_E, A_I = _phase_plan(model, time, current_func, condition)
		figure.add_trace(
			go.Scatter(
				x=A_E,
				y=A_I,
				mode='lines',
				name=f'({condition[0]}, {condition[1]})'
			)
		)
	arr_nulcline_ae = np.linspace(0, 0.2, 500)
	arr_nulcline_ai = np.linspace(0, 0.1, 500)
	nulcline_ae = model.nullcline_A_E(arr_nulcline_ae, current_func(0)[0])
	nulcline_ai = model.nullcline_A_I(arr_nulcline_ai, current_func(0)[1])
	figure.add_trace(
		go.Scatter(
			x=arr_nulcline_ae[~np.isnan(nulcline_ae)],
			y=nulcline_ae[~np.isnan(nulcline_ae)],
			mode='lines',
			name='nullcline A<sub>E</sub>',
			line_width=3,
			line_color='orange'
		)
	)
	figure.add_trace(
		go.Scatter(
			x=nulcline_ai[~np.isnan(nulcline_ai)],
			y=arr_nulcline_ai[~np.isnan(nulcline_ai)],
			mode='lines',
			name='nullcline A<sub>I</sub>',
			line_width=3,
			line_color='crimson'
		)
	)
	figure = phase_plan_layout(figure)
	figure.show()


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
	phase_plan_var_init(weights, I_func)
