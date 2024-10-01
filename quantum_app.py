from dash import Dash, html, dcc, Output, Input
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from helper_funcs import variational_regressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Advertising.csv')
TV_ads = df['TV'].values
Total_sales = df['Sales'].values

x_scaler = StandardScaler()
y_scaler = StandardScaler()
X = x_scaler.fit_transform(TV_ads.reshape(-1, 1)).flatten()
Y = y_scaler.fit_transform(Total_sales.reshape(-1, 1)).flatten()

x_scale = x_scaler.scale_[0]
y_scale = y_scaler.scale_[0]
weight_lists = np.load('weights.npy')
epoch_values = [0] + [i for i in range(9, 50, 10)] + [99, 149]
app = Dash(suppress_callback_exceptions=True)

app.layout = html.Div(
    children=[
        html.H1(children='Advertising Analysis', style={'textAlign':'center'}),
        html.Div([
            dcc.Graph(id='graph-output'),
            dcc.Slider(
                id='weight-slider',
                min=min(epoch_values),
                max=max(epoch_values),
                step=None, 
                value=epoch_values[0],
                marks={epoch: str(epoch) for epoch in epoch_values},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ]),
        # html.Div(id='inputs-container', style={'display': 'none'}),
        html.P("Epoch #", style={'text-align': 'center', 'font-weight': 'bold'})
    ]
)
@app.callback(
    Output('graph-output', 'figure'),
    [Input('weight-slider', 'value')],
    prevent_initial_callback=True
)
def update_graph(epoch_value):
    index = epoch_values.index(epoch_value)
    weight_list = weight_lists[index]
    original_X = x_scaler.inverse_transform(np.linspace(-2, 2, 100).reshape(-1, 1)).flatten()

    predicted_Y = [variational_regressor(x, weight_list) for x in np.linspace(-2, 2, 100)]
    unnormalized_Y = y_scaler.inverse_transform(np.array(predicted_Y).reshape(-1, 1)).flatten()
    
    
    fig = px.scatter(x=x_scaler.inverse_transform(X.reshape(-1, 1)).flatten(),
                      y=y_scaler.inverse_transform(Y.reshape(-1, 1)).flatten())
    fig.add_trace(go.Scatter(x=original_X, 
                            y=unnormalized_Y,
                        mode='lines',
                        line=dict(color='red'),
                        name='Fitted Line'
    ))
    fig.update_layout(
        title={
            'text': f"Quantum TV Predictions Per Epoch",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='TV Ads (in thousands)',
        yaxis_title='Total Sales (in millions)'
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)
