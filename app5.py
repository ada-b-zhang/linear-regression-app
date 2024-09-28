from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import numpy as np
import pandas as pd


app = Dash(__name__)


intercepts = {i:i for i in range(-20, 21)}

slopes = {i:i for i in range(-10, 11)}

n = {i:i for i in range(101)}


app.layout = html.Div([
    html.H1('Make Your Own Scatterplot!', 
            style={'textAlign': 'center'}),
    html.H4('Play around with the intercept, slope, and/or sample size, and notice how the plot behaves. Do you notice how the regression line changes? Do you notice anything about the dotted red lines?', 
            style={'textAlign': 'center'}),
    dcc.Graph(id="graph"),

    html.P("Select intercept:"),
    dcc.Dropdown(
        id='intercept_dropdown',
        options=[{'label': i, 'value': i} for i in intercepts.values()],
        value=0,
        clearable=False),

    html.P("Select slope:"),
    dcc.Dropdown(
        id='slope_dropdown',
        options=[{'label': i, 'value': i} for i in slopes.values()],
        value=0,
        clearable=False),

    html.P("Select sample size:"),
    dcc.Dropdown(
        id='n_dropdown',
        options=[{'label': i, 'value': i} for i in n.values()],
        value=10,
        clearable=False)
])


def generate_data(beta0, beta1, n, sigma=2):
    X = np.linspace(start=-10, stop=10, num=n).reshape(-1, 1) # array w/ 1 column, the -1 means 'infer the # of rows'
    
    # Generate epsilon as a 1D array of N(0, sigma^2) RVs
    epsilon = sigma * np.random.randn(n)
    
    # Generate y using SLR model
    y = beta0 + (beta1 * X.flatten()) + epsilon

    return X.flatten(), y, X.mean(), y.mean()


@app.callback(
    Output("graph", "figure"), 
    Input("intercept_dropdown", "value"), 
    Input("slope_dropdown", "value"),
    Input("n_dropdown", "value"))
def display_color(intercept_dropdown, slope_dropdown, n_dropdown):
    x_data, y_data, x_mean, y_mean = generate_data(intercept_dropdown, slope_dropdown, n_dropdown)
    fig = px.scatter(x=x_data, y=y_data, trendline='ols', color_discrete_sequence=["black"])
    fig.update_traces(line=dict(color="black"))
    fig.add_vline(x=x_mean, line_dash="dash", line_color="red")
    fig.add_hline(y=y_mean, line_dash="dash", line_color="red")
    fig.data[0].showlegend = False
    return fig

app.run_server(debug=True)

