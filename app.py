from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

advertising_df = pd.read_csv('advertising.csv')
advertising_df = advertising_df.drop('Unnamed: 0', axis=1)
app = Dash(suppress_callback_exceptions=True)

app.layout = html.Div(
    children=[
        html.H1(children='Advertising Analysis', style={'textAlign':'center'}),
        dcc.Dropdown(
            advertising_df.columns[:-1], 'TV', 
            id='dropdown-selection',
        ),
        html.Div([(html.Div([dcc.Graph(id='graph-content')], style={'width': '50%', 'height': '200%'})), 
        (html.Div([dcc.Graph(id='graph-content-2'),
        dcc.Graph(id='graph-content-3')], style={'width': '50%'}))], style={'display': 'flex'}),
        html.Div(id='inputs-container', style={'display': 'none'}),
        html.P(children="This is our amazing blog post.")
    ]
)

@app.callback(
    Output('inputs-container', 'children'),
    [Input('dropdown-selection', 'value')]
)
def update_inputs(selected_column):
    # Get the columns that are not selected
    cols_to_avoid = [selected_column, 'Sales']
    remaining_columns = [col for col in advertising_df.columns if col not in cols_to_avoid]
    # print(selected_column)
    # print("Remaining Columns: ", remaining_columns)
    # Create two inputs for the remaining columns
    inputs = [
            dcc.Store(id='sub-input-1', data=remaining_columns[0]),
            dcc.Store(id='sub-input-2', data=remaining_columns[1])
        ]

    return inputs


@callback(
    Output('graph-content', 'figure'),
    [Input('dropdown-selection', 'value'),
     Input('graph-content', 'hoverData')],
    prevent_initial_callback=True
)
def update_graph(value, hover_data):
    # marker_opacity = [1.0] * len(advertising_df)
    colors = ['blue'] * len(advertising_df)
    sizes = [5] * len(advertising_df)
    if hover_data:
        hover_idx = hover_data['points'][0]['pointIndex']
        # marker_opacity[hover_idx] = 1.0
        colors[hover_idx] = 'red'
        sizes[hover_idx] = 10
    fig = px.scatter(advertising_df, x=value, y='Sales', trendline="ols")
    fig.update_layout(
        title={
            'text': f"{value} Impact on Sales",
            'x': 0.5,
            'xanchor': 'center'
        },
        hovermode='closest'
    )
    fig.update_traces(
            selector=dict(mode='markers'),
            marker=dict(
                # opacity=marker_opacity,
                color=colors,
                size=sizes,
                line=dict(width=0)
            ),
            
            hoverinfo='text'
        )
    
    fig.update_traces(
        line=dict(color="red"),
        selector=dict(mode="lines"),
    )
    return fig

@callback(
    Output('graph-content-2', 'figure'),
   [Input('sub-input-1', 'data'),
    Input('graph-content', 'hoverData')],
    prevent_initial_call=True
)
def update_graph_2(value, hover_data):
    colors = ['blue'] * len(advertising_df)
    sizes = [5] * len(advertising_df)
    hover_idx = 0
    hover_text = ''
    if hover_data:
        hover_idx = hover_data['points'][0]['pointIndex']
        hover_text = advertising_df.iloc[0][value]
        # marker_opacity[hover_idx] = 1.0
        colors[hover_idx] = 'red'
        sizes[hover_idx] = 10
        
    fig = px.scatter(advertising_df, x=value, y='Sales', trendline="ols")
    fig.update_layout(
        title={
            'text': f"{value} Impact on Sales",
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    fig.update_traces(
            selector=dict(mode='markers'),
            marker=dict(
                # opacity=marker_opacity,
                color=colors,
                size=sizes,
                line=dict(width=0)
            ),
            hoverinfo='text',
            hovertext=[hover_text if i == hover_idx else '' for i in range(len(advertising_df))]
        )
    fig.update_traces(
        line=dict(color="red"),
        selector=dict(mode="lines")
    )
    
    return fig

@callback(
    Output('graph-content-3', 'figure'),
    [Input('sub-input-2', 'data'),
     Input('graph-content', 'hoverData')],
    prevent_initial_call=True
)
def update_graph_3(value, hover_data):
    # marker_opacity = [0.5] * len(advertising_df)
    colors = ['blue'] * len(advertising_df)
    sizes = [5] * len(advertising_df)
    hover_idx = 0
    hover_text = ''
    if hover_data:
        hover_idx = hover_data['points'][0]['pointIndex']
        hover_text = advertising_df.iloc[0][value]
        # marker_opacity[hover_idx] = 1.0
        colors[hover_idx] = 'red'
        sizes[hover_idx] = 10
    fig = px.scatter(advertising_df, x=value, y='Sales', trendline="ols")
    fig.update_layout(
        title={
            'text': f"{value} Impact on Sales",
            'x': 0.5,
            'xanchor': 'center'
        }
    )

    fig.update_traces(
            selector=dict(mode='markers'),
            marker=dict(
                # opacity=marker_opacity,
                color=colors,
                size=sizes,
                line=dict(width=0)
            ),
            hoverinfo='text',
            hovertext=[hover_text if i == hover_idx else '' for i in range(len(advertising_df))]
        )

    fig.update_traces(
        line=dict(color="red"),
        selector=dict(mode="lines")
    )
    
    return fig

if __name__ == '__main__':
    app.run(debug=True)
