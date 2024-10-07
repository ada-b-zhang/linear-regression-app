from dash import Dash, html, dcc, Output, Input
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from helper_funcs import variational_regressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Advertising.csv')
TV_ads = df['TV'].values
Radio_ads = df['Radio'].values
Newspaper_ads = df['Newspaper'].values
Total_sales = df['Sales'].values

tv_scaler = StandardScaler()
radio_scaler = StandardScaler()
newspaper_scaler = StandardScaler()
y_scaler = StandardScaler()
scaler_map = {'TV': tv_scaler, 'Radio': radio_scaler, 'Newspaper': newspaper_scaler}

TV_X = tv_scaler.fit_transform(TV_ads.reshape(-1, 1)).flatten()
Radio_X = radio_scaler.fit_transform(Radio_ads.reshape(-1, 1)).flatten()
Newspaper_X = newspaper_scaler.fit_transform(Newspaper_ads.reshape(-1, 1)).flatten()
Y = y_scaler.fit_transform(Total_sales.reshape(-1, 1)).flatten()
X_map = {'TV': TV_X, 'Radio': Radio_X, 'Newspaper': Newspaper_X}

linspace_map = {'TV': np.linspace(-2, 2, 100), 'Radio': np.linspace(-2, 2, 100), 'Newspaper': np.linspace(-2, 4, 100)}


# x_scale = x_scaler.scale_[0]
# y_scale = y_scaler.scale_[0]
tv_weights = np.load('TV-weights.npy')
radio_weights = np.load('Radio-weights.npy')
newspaper_weights= np.load('Newspaper-weights.npy')
weights_map = {'TV': tv_weights, 'Radio': radio_weights, 'Newspaper': newspaper_weights}

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
            ),
            html.P("Epoch #", style={'text-align': 'center', 'font-weight': 'bold'}),
            dcc.Dropdown(
                id='dropdown-input',
                options=[
                    {'label': 'TV', 'value': 'TV'},
                    {'label': 'Radio', 'value': 'Radio'},
                    {'label': 'Newspaper', 'value': 'Newspaper'},
                ],
                value='TV',
                clearable=False
            )
        ]),
    ]
)
@app.callback(
    Output('graph-output', 'figure'),
    [Input('weight-slider', 'value'),
     Input('dropdown-input', 'value')],
    prevent_initial_callback=True
)
def update_graph(epoch_value, ad_type):
    index = epoch_values.index(epoch_value)
    weight_lists = weights_map[ad_type]
    weight_list = weight_lists[index]
    x_scaler = scaler_map[ad_type]
    X = X_map[ad_type]
    linspace = linspace_map[ad_type]
    original_X = x_scaler.inverse_transform(linspace.reshape(-1, 1)).flatten()

    predicted_Y = [variational_regressor(x, weight_list) for x in linspace]
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
            'text': f"Quantum {ad_type} Predictions Per Epoch",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title=f'{ad_type} Ads (in thousands)',
        yaxis_title='Total Sales (in millions)'
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)
