from question1 import WilsonCowanModel, display_surface_wee_time, display_surfaces_WC
from dash import Dash, html, dcc
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
	"width": "16rem",
	"padding": "2rem 1rem",
	"background-color": "#f8f9fa",
}

CONTENT_STYLE = {
	"margin-left": "18rem",
	"margin-right": "2rem",
	"padding": "2rem 1rem",
}


def make_sidebar():
	parameters_I, parameters_W = ['I_E', 'I_I'], ['W_EE', 'W_EI', 'W_IE', 'W_II']
	axis_dropdown = html.Div(
		[
			dbc.Label("Choose y axis"),
			dbc.Row(
				dbc.Col(
					dcc.Dropdown(
					placeholder="I_E",
					value="I_E",
					id="axis-dropdown",
					options=parameters_I+parameters_W
					)
				),
				no_gutters=True
			),
		]
	)

	for param in parameters_I:
		pass

	return html.Div(
		[
			html.H2("Sidebar", className="display-4"),
			html.Hr(),
			dbc.Row(axis_dropdown)
		],
		style=SIDEBAR_STYLE,
	)


content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div(
	[
		make_sidebar(),
		content
	]
)
# Callback
