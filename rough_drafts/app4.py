from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import numpy as np

app = Dash(__name__)


app.layout = html.Div([
    html.H1('Visualizing the Gaussian (Normal) Distribution', 
            style={'textAlign': 'center'}),
    html.H4('Try changing the mean, standard deviation, and/or sample size. How does the distribution change?', 
            style={'textAlign': 'center'}),
    dcc.Graph(id="graph"),

    html.P("Mean:"),
    dcc.Slider(id="mean", min=-5, max=5, value=0, 
               marks={-5: '-5', -4:'-4', -3: '-3', -2:'-2', -1:'-1', 0:'0',
                      1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}),

    html.P("Standard Deviation:"),
    dcc.Slider(id="std", min=1, max=3, value=1, 
               marks={1: '1', 2: '2', 3: '3'}),

    html.P("Sample Size:"),
    dcc.Slider(id="sample_size", min=0, max=10000, step=100, value=500,
               marks={0: '0', 1000: '1000', 2000: '2000', 3000: '3000', 4000: '4000', 5000:'5000',
                      6000:'6000',  7000:'7000',  8000:'8000',  9000:'9000', 10000:'10000'}),
])


@app.callback(
    Output("graph", "figure"), 
    Input("mean", "value"), 
    Input("std", "value"),
    Input("sample_size", "value"))
def display_color(mean, std, sample_size):
    data = np.random.normal(mean, std, size=sample_size) 
    fig = px.histogram(data, range_x=[-10, 10], nbins=50, color_discrete_sequence=['purple'])
    fig.data[0].showlegend = False
    return fig

app.run_server(debug=True)


# References: https://plotly.com/python/histograms/
