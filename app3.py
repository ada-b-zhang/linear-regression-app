import dash
from dash import dcc, html
import plotly.express as px
import numpy as np
from skimage import io
import pandas as pd

advertising_data = pd.read_csv('advertising.csv')  
advertising_data = advertising_data.drop('Unnamed: 0', axis=1)

# Initialize the Dash app
app = dash.Dash(__name__)

fig1 = px.scatter(advertising_data, 
                 x="TV", y="Sales", size='Newspaper',
                 color='Radio', trendline="ols",
                 title='Another way to see effects of several predictors on response:')
# Create the image plot 
# img_url = 'pic.png'
# img = io.imread(img_url)
# fig2 = px.imshow(img)    

# app.layout = html.Div([dcc.Graph(id='scatter-plot',figure=fig1),
#                        dcc.Graph(id='image-plot',figure=fig2)])
app.layout = html.Div([dcc.Graph(id='scatter-plot',figure=fig1)])

if __name__ == '__main__':
    app.run_server(debug=True)

# Resources:
# https://plotly.com/python/imshow/
# https://plotly.com/python/linear-fits/
