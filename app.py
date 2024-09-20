from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

advertising_df = pd.read_csv('advertising.csv')
advertising_df = advertising_df.drop('Unnamed: 0', axis=1)
app = Dash()

app.layout = html.Div(
    children=[
        html.H1(children='Advertising Analysis', style={'textAlign':'center'}),
        dcc.Dropdown(
            advertising_df.columns[:-1], 'TV', 
            id='dropdown-selection',
            style={
                'backgroundColor': 'black',  # Change dropdown background color
                'color': 'white'  # Change text color
            }
            # # This changes the background and text color of the options
            # dropdown_style={
            #     'backgroundColor': 'black',  # Dropdown options background color
            #     'color': 'white'  # Dropdown options text color
            # }
        ),
        dcc.Graph(id='graph-content'),
        html.P(children="This is our amazing blog post.")
    ]
)

@callback(
    Output('graph-content', 'figure'),
    Input('dropdown-selection', 'value')
)
def update_graph(value):
    # dff = advertising_df[value]
    fig = px.scatter(advertising_df, x=value, y='Sales', trendline="ols")
    fig.update_layout(
        title={
            'text': f"{value} Impact on Sales",
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(color="white")
        },
        xaxis=dict(
            title=dict(text=f"{value} Ads in $1000", font=dict(color='white')),
            tickfont=dict(color='white')
        ),
        yaxis=dict(
            title=dict(text="Sales Revenue in Millions", font=dict(color='white')),
            tickfont=dict(color='white')
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
    )

    fig.update_traces(
        line=dict(color="red"),
        selector=dict(mode="lines")
    )
    
    return fig

if __name__ == '__main__':
    app.run(debug=True)
