from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd

app = Dash(__name__)
advertising_df = pd.read_csv('advertising.csv')  
advertising_df = advertising_df.drop('Unnamed: 0', axis=1)
min_TV = min(advertising_df['TV'])
max_TV = max(advertising_df['TV'])

app.layout = html.Div([
    html.H1('Visualizing Multiple Predictors', 
            style={'textAlign': 'center'}),
    html.H4('The impact of Radio and Newspaper on Sales. Feel free to filter to certain values of TV!', 
            style={'textAlign': 'center'}),    
    dcc.Graph(id="3D_graph"),
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
    Output("3D_graph", "figure"), 
    Input("range-slider", "value"))
def update_bar_chart(slider_range):
    low, high = slider_range
    mask = (advertising_df['TV'] >= low) & (advertising_df['TV'] <= high)

    fig_3D_graph = px.scatter_3d(advertising_df[mask], 
        x='Radio', 
        y='Newspaper', 
        z='Sales',
        hover_data=['TV'])

    fig_3D_graph.update_traces(marker=dict(color='purple'))  
    return fig_3D_graph

if __name__ == '__main__':
    app.run_server(debug=True)


# Resources: https://plotly.com/python/3d-scatter-plots/
