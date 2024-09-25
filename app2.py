from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd

app = Dash(__name__)
advertising_data = pd.read_csv('advertising.csv')  
advertising_data = advertising_data.drop('Unnamed: 0', axis=1)
min_TV = min(advertising_data['TV'])
max_TV = max(advertising_data['TV'])

app.layout = html.Div([
    html.H4('not sure if im gonna finish this project or if this project is gonna finish me'),
    dcc.Graph(id="graph"),
    html.P("TV"),
    dcc.RangeSlider(
        id='range-slider',
        min=0,  
        max=300,
        step=30,  
        marks={i: str(i) for i in range(0, 301, 30)},
        value=[0, 300] 
    ),
])

@app.callback(
    Output("graph", "figure"), 
    Input("range-slider", "value"))
def update_bar_chart(slider_range):
    low, high = slider_range
    mask = (advertising_data['TV'] >= low) & (advertising_data['TV'] <= high)

    fig = px.scatter_3d(advertising_data[mask], 
        x='Radio', 
        y='Newspaper', 
        z='Sales',
        hover_data=['TV'])

    fig.update_traces(marker=dict(color='purple'))  
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)


# Resources: https://plotly.com/python/3d-scatter-plots/
