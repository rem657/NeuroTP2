import numpy as np

from question1 import WilsonCowanModel, display_surface_wee_time, display_surfaces_WC, go, display_surface
from dash import Dash, html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc

# APP
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)
server = app.server

# Layout
SIDEBAR_STYLE = {
	"position": "fixed",
	"top": 0,
	"left": 0,
	"bottom": 0,
	"width": "25rem",
	"padding": "2rem 1rem",
	"background-color": "#f8f9fa",
	'color': 'black'
}

CONTENT_STYLE = {
	"margin-left": "25rem",
	"margin-right": "2rem",
	"padding": "2rem 1rem",
}

pointer_id = {'I_E': 'I_E', 'I_I': 'I_I', 'W_EE': 'wee', 'W_EI': 'wei', 'W_IE': 'wie', 'W_II': 'wii'}
PARAMETER = list(pointer_id.keys())


def make_sidebar():
	axis_dropdown = html.Div(
		[
			dbc.Label("Choose y axis"),
			dbc.Row(
				dbc.Col(
					dcc.Dropdown(
						placeholder='Parameter',
						value=PARAMETER[0],
						id="axis-dropdown",
						options=PARAMETER
					)
				),
			),
		],
		style={'margin-top': '3rem'}
	)
	y_axis_selector = html.Div(
		[
			dbc.Row(
				dbc.Col(
					[
						dbc.Label(f"Choose y axis range: "),
						dbc.Row(
							[
								dbc.Col(
									[
										dbc.Input(type="number", value=0, step=0.5, inputmode='numeric', id='ymin'),
										dbc.FormText("Min")
									],
									className="me-3"),
								dbc.Col(
									[
										dbc.Input(type="number", value=60, step=0.5, inputmode='numeric', id='ymax'),
										dbc.FormText("Max")
									],
									className="me-3"),
								dbc.Col(
									[
										dbc.Input(type="number", value=0.5, step=0.5, min=0, inputmode='numeric', id='ystep'),
										dbc.FormText("Step size")
									],
									className="me-3")
							],
							className='g-1'
						)
					]
				),
			)
		],
		style={'margin-top': '3rem'}
	)
	weight_selection = html.Div(
		[
			dbc.Label(f"Choose weights :"),
			dbc.Row(
				[
					dbc.Col(
						[
							dbc.Input(type="number", step=0.5, min=0, value=0.0, inputmode='numeric', id='wee'),
							dbc.FormText(["W", html.Sub('EE')])
						],
						className="me-3"),
					dbc.Col(
						[
							dbc.Input(type="number", step=0.5, min=0, value=0.0, inputmode='numeric', id='wei'),
							dbc.FormText(["W", html.Sub('EI')])
						],
						className="me-3")
				],
				className='g-1'
			),
			dbc.Row(
				[
					dbc.Col(
						[
							dbc.Input(type="number", step=0.5, min=0, value=0.0, inputmode='numeric', id='wie'),
							dbc.FormText(["W", html.Sub('IE')])										],
						className="me-3"),
					dbc.Col(
						[
							dbc.Input(type="number", step=0.5, min=0, value=0.0, inputmode='numeric', id='wii'),
							dbc.FormText(["W", html.Sub('II')])										],
						className="me-3")
				],
				className='g-1'
			),
		],
		style={'margin-top': '3rem'}
	)
	current_selector = html.Div(
		dbc.Col(
			[
				dbc.Label(f"Choose current: "),
				dbc.Row(
					[
						dbc.Col(
							[
								dbc.Input(type="number", step=0.5, value=0, inputmode='numeric', id='I_E'),
								dbc.FormText(["I", html.Sub('E')])
							],
							className="me-3"),
						dbc.Col(
							[
								dbc.Input(type="number", step=0.5, value=0, inputmode='numeric', id='I_I'),
								dbc.FormText(["I", html.Sub('I')])
							],
							className="me-3")
					],
					className='g-1'
				)
			]
		),
		style={'margin-top': '3rem'}
	)
	integrate_button = html.Div(
		dbc.Button("Integrate", color="primary", id='main-button'),
		className="d-grid gap-2 col-10 mx-auto",
		style={'margin-top': '5rem'}
	)
	save_button = html.Div(
		[
			dbc.Button("Save Figure", color="warning", id='save-button'),
			dbc.Toast(
				[html.P("Figure saved !", className="mb-0")],
				id="save-toast",
				color='dark',
				header="You did it!!!",
				dismissable=True,
				is_open=False,
				duration=2000,
				body_style={'color': "white"}
			),
		],
		className="d-grid gap-2 col-10 mx-auto",
		style={'margin-top': '5rem'}
	)
	return html.Div(
		[
			html.H2("Menu", className="display-4"),
			html.Hr(),
			dbc.Row(axis_dropdown),
			dbc.Row(y_axis_selector),
			dbc.Row(weight_selection),
			dbc.Row(current_selector),
			dbc.Row(integrate_button),
			dbc.Row(save_button)
		],
		style=SIDEBAR_STYLE,
	)


content = html.Div(
	dcc.Graph(
		figure=go.Figure(),
		id='graph',
		style={'height': '95vh'}
	),
	id="page-content",
	style=CONTENT_STYLE
)

app.layout = html.Div(
	[
		make_sidebar(),
		content
	]
)


# Callback
@app.callback(
	Output('I_E', 'disabled'),
	Output('I_I', 'disabled'),
	Output('wee', 'disabled'),
	Output('wei', 'disabled'),
	Output('wie', 'disabled'),
	Output('wii', 'disabled'),
	Input("axis-dropdown", 'value')
)
def deactivate_active_axis(active_axis: str):
	axis_index = PARAMETER.index(active_axis)
	deactivate = np.zeros((len(pointer_id),), dtype=bool)
	deactivate[axis_index] = True
	return tuple(deactivate)


@app.callback(
	Output('graph', 'figure'),
	Input('main-button', 'n_clicks'),
	State('axis-dropdown', 'value'),
	State('ymin', 'value'),
	State('ymax', 'value'),
	State('ystep', 'value'),
	State('I_E', 'value'),
	State('I_I', 'value'),
	State('wee', 'value'),
	State('wei', 'value'),
	State('wie', 'value'),
	State('wii', 'value')
)
def update_graph(n_clicks: int, axis_val, min_axis, max_axis, step_size, I_E, I_I, W_ee, W_ei, W_ie, W_ii):
	if n_clicks is not None:
		arr_variable = np.arange(min_axis, max_axis, step_size)
		if axis_val == 'I_E':
			I_E = arr_variable
		elif axis_val == 'I_I':
			I_I = arr_variable
		elif axis_val == 'W_EE':
			W_ee = arr_variable
		elif axis_val == 'W_EI':
			W_ei = arr_variable
		elif axis_val == 'W_IE':
			W_ie = arr_variable
		elif axis_val == 'W_II':
			W_ii = arr_variable
		figure = display_surface(I_E, I_I, W_ee, W_ei, W_ie, W_ii, 0, 100, 0.2, 0.2, 1.0, 0.2)
		figure.update_layout(
			paper_bgcolor='rgba(0, 0, 0, 0)',
			legend=dict(
				bgcolor='white',
				borderwidth=5,
			),
			scene=dict(
				xaxis=dict(
					color='grey',
				),
				yaxis=dict(
					color='grey',
				),
				zaxis=dict(
					color='grey',
				),
			)
		)
		return figure
	else:
		return no_update


@app.callback(
	Output("save-toast", "is_open"),
	Input('save-button', 'n_clicks'),
	State('graph', 'figure'),
	State('axis-dropdown', 'value'),
	State('ymin', 'value'),
	State('ymax', 'value'),
	State('I_E', 'value'),
	State('I_I', 'value'),
	State('wee', 'value'),
	State('wei', 'value'),
	State('wie', 'value'),
	State('wii', 'value')
)
def save_figure(n_clicks, figure, axis_val, ymin, ymax, I_E, I_I, W_EE, W_EI, W_IE, W_II):
	if n_clicks is not None:
		figure_title = f'surface_{axis_val}_{ymin=}_{ymax=}_'
		if axis_val != 'I_E':
			figure_title += f'{I_E=}_'
		if axis_val != 'I_I':
			figure_title += f'{I_I=}_'
		if axis_val != 'W_EE':
			figure_title += f'{W_EE=}_'
		if axis_val != 'W_EI':
			figure_title += f'{W_EI=}_'
		if axis_val != 'W_IE':
			figure_title += f'{W_IE=}_'
		if axis_val != 'W_II':
			figure_title += f'{W_II=}_'
		figure_title += '.html'
		go.Figure(**figure).write_html(figure_title)
		return True
	else:
		return no_update


if __name__ == '__main__':
	app.run_server(debug=True)
