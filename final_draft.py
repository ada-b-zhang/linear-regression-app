########################################################################################################################################################################################################################################################################################################################
# IMPORTING PACKAGES
########################################################################################################################################################################################################################################################################################################################
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Output, Input
from dash.dependencies import Input, Output
from helper_funcs import variational_regressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression



########################################################################################################################################################################################################################################################################################################################
# STUFF FOR THE GRAPHS
########################################################################################################################################################################################################################################################################################################################
"""
These are just some variables and data needed to create the graphs. 
"""

# Dataset
advertising_df = pd.read_csv('Advertising.csv')
advertising_df = advertising_df.drop('Unnamed: 0', axis=1)

# Prepare variables and data
TV_ads = advertising_df['TV'].values
Radio_ads = advertising_df['Radio'].values
Newspaper_ads = advertising_df['Newspaper'].values
Total_sales = advertising_df['Sales'].values

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
epoch_values = [0] + [i for i in range(9, 50, 10)] + [99, 149]

# Load the pretrained weights for each predictor
tv_weights = np.load('TV-weights.npy')
radio_weights = np.load('Radio-weights.npy')
newspaper_weights= np.load('Newspaper-weights.npy')
weights_map = {'TV': tv_weights, 'Radio': radio_weights, 'Newspaper': newspaper_weights}

# For "Make Your Own Scatterplot" graph
intercepts = {i: i for i in range(-20, 21)}
slopes = {i: i for i in range(-10, 11)}
n = {i: i for i in range(101)}
def generate_data(beta0, beta1, n, sigma=2):
    X = np.linspace(start=-10, stop=10, num=n).reshape(-1, 1)  # array w/ 1 column
    epsilon = sigma * np.random.randn(n)  # generate epsilon
    y = beta0 + (beta1 * X.flatten()) + epsilon  # generate y using SLR model
    return X.flatten(), y, X.mean(), y.mean()



########################################################################################################################################################################################################################################################################################################################
# INITIALIZE DASH APP
########################################################################################################################################################################################################################################################################################################################
app = Dash(
    suppress_callback_exceptions=True,
    external_stylesheets=[
        'https://fonts.googleapis.com/css2?family=Lora:wght@400;700&family=Montserrat:wght@300&display=swap',
        'https://fonts.googleapis.com/css2?family=Sixtyfour+Convergence&display=swap'])



########################################################################################################################################################################################################################################################################################################################
# CREATING DROPDOWN 
########################################################################################################################################################################################################################################################################################################################
# Create dropdown options by mapping column names to custom labels
column_labels = {
    'TV': 'TV Advertising',
    'Radio': 'Radio Advertising',
    'Newspaper': 'Newspaper Advertising'}
dropdown_options = [{'label': column_labels[col], 'value': col} for col in advertising_df.columns[:-1]]



########################################################################################################################################################################################################################################################################################################################
# CREATING THE LAYOUT
########################################################################################################################################################################################################################################################################################################################
"""
There are comments to help indicate what each chunk of code does. 
If you are on a mac, you can directly type emojis!

"""

app.layout = html.Div(
    children=[

        # TITLE & SUBTITLE
        html.Header(
            children=[
                html.H1("Linear Regression Demonstration", 
                                                    style={'font-family': 'Lora, serif',
                                                            # 'font-weight': 'bold',
                                                            'font-weight': '800',
                                                            'font-size': '4.0rem',
                                                            'color': 'blue',
                                                            'textAlign': 'center',
                                                            'margin-bottom': '20px'}),
                html.P("A Lesson and Demonstration on Linear Regression for any Audience", 
                                                    style={'font-family': 'Montserrat, sans-serif',
                                                            'font-size': '2.0rem',
                                                            'textAlign': 'center',
                                                            'color': 'grey',
                                                            'margin-top': '30px',
                                                            'margin-bottom': '60px'})
            ]
        ),
        # I. INTRODUCTION ############################################################################################
        html.Div(children=[

        html.H2("I. Introduction 😃", 
                                            style={'font-family': 'Lora, serif',
                                            'font-weight': '700',
                                            'font-size': '2.5rem',
                                            'color': '#2c3e50',
                                            'textAlign': 'center',
                                            # 'margin-left': '200px',
                                            'margin-bottom': '20px',
                                            'margin-top': '40px'}),
        
        html.P(["Have you ever wondered how interconnected the world is? Many things around us seem to have relationships with each other. For example, wingspan and height, caffeine intake and energy levels, study hours and exam scores, and money spent on advertisements and profit earned. However, is there a way we can quantify and visualize these relationships? More importantly, is there a way to empirically show the relationships around us? If only there were a statistical tool we can use to help us do these things…", 
                        html.Br(), html.Br(),
                        "An answer is ", html.B("linear regression"), ", which helps to quantify the relationship between ", html.B("quantitative (numeric)"), " variables.",
                        html.Br(), html.Br(),
                        "To explain, think about a farmer selling oranges at a farmers market. Each orange that the farmer sells means more money earned. Let's denote the number of oranges sold as X and the profit earned as Y. In simple linear regression, X is the ", html.B("predictor"), " variable, and Y is the ", html.B("response"), " variable. Simple linear regression aims to quantify and estimate the relationship between X and Y. In our example, let's say that each orange costs $2. Therefore, we know that for each orange that the farmer sells, he makes $2. Or, to put this into statistical language, as X increases by 1, Y increases by 2. This is the essence of linear regression: how much the response Y increases/decreases as the predictor X increases/decreases.",
                        html.Br(), html.Br(),
                        "Note: one orange for $2 is quite expensive! 🍊"], 
                        style={'font-family': 'Lora, serif',
                                'font-size': '1.25rem',
                                'color': '#2c3e50',
                                'textAlign': 'left',
                                'max-width': '1000px',
                                'margin': 'auto',
                                'margin-bottom': '20px'}),
        # II. LESSON AND ILLUSTRATION OF THE BASICS ############################################################################################
        html.H2("II. Lesson and Illustration of the Basics 🤓", 
                                                style={'font-family': 'Lora, serif',
                                                'font-weight': '700',
                                                'font-size': '2.5rem',
                                                'color': '#2c3e50',
                                                'textAlign': 'center',
                                                'margin-bottom': '20px',
                                                'margin-top': '40px'}),

        # Heading for Linear Regression Explanation
        html.H2("How does Classical Linear Regression work?", 
                                                style={'font-family': 'Montserrat, sans-serif',
                                                        'font-size': '2.0rem',
                                                        'textAlign': 'center',
                                                        'color': 'black',
                                                        'margin-top': '30px'
                                                        }),
        # Normal explanation of Linear Regression
        html.P("Linear regression is one of the simplest and most widely used predictive modeling techniques. It models the relationship between a dependent variable and one or more independent variables. The objective is to find a linear equation that best fits the data points. This technique assumes that the relationship between the dependent and independent variables is linear, this is, a straight line can be drawn through the data points to find a relationship.", 
                       style={'font-family': 'Lora, serif',
                                'font-size': '1.25rem',
                                'color': '#2c3e50',
                                'textAlign': 'left',
                                'max-width': '1000px',
                                'margin': 'auto',
                                'margin-bottom': '20px'}),

        # Mathematical explanation of Linear Regression
        html.P("In mathematical terms, the linear regression model can be represented as:", 
                           style={'font-family': 'Lora, serif',
                                'font-size': '1.25rem',
                                'color': '#2c3e50',
                                'textAlign': 'left',
                                'max-width': '1000px',
                                'margin': 'auto',
                                'margin-bottom': '20px'
        }),
        dcc.Markdown("""
        $$
        y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + \dots + \\beta_n x_n + \epsilon
        $$
        """, 

                style={'font-family': 'Lora, serif',
                    'font-size': '1.25rem',
                    'color': '#2c3e50',
                    'textAlign': 'left',
                    'max-width': '1000px',
                    'margin': 'auto',
                    'margin-bottom': '20px'}, mathjax=True),

        html.P("Where:", 
                       style={'font-family': 'Lora, serif',
                                'font-size': '1.25rem',
                                'color': '#2c3e50',
                                'textAlign': 'left',
                                'max-width': '1000px',
                                'margin': 'auto',
                                'margin-bottom': '20px'
        }),
        dcc.Markdown("""
        - $y$ is the **dependent** variable (the outcome we are trying to predict)
        - $x_1, x_2, ..., x_n$ are the **independent** variables (the predictors)
        - $\\beta_0$ is the **intercept** (the value of $y$ when all $x_i$ are zero)
        - $\\beta_1, \\beta_2, ..., \\beta_n$ are the **coefficients** (the impact of each $x_i$ on $y$)
        - $\\epsilon$ is the **error** term (the difference between the observed and predicted values)
        """, style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'line-height': '1.5'
        }, mathjax=True),

        html.H2("Assumptions of Regression", 
                            style={'font-family': 'Montserrat, sans-serif',
                                    'font-size': '2.0rem',
                                    'textAlign': 'center',
                                    'color': 'black',
                                    'margin-top': '30px'}),

        html.P(["There are some ", html.B("assumptions"), " we need to make in order to perform linear regression:"], 
                           style={'font-family': 'Lora, serif',
                                'font-size': '1.25rem',
                                'color': '#2c3e50',
                                'textAlign': 'left',
                                'max-width': '1000px',
                                'margin': 'auto',
                                'margin-bottom': '20px'
        }),
        dcc.Markdown("""
        - **Linearity**: X and Y have a linear relationship with each other 
        - **Normality**: the errors follow a Normal (Gaussian) distribution
        - **Constant Variance**: also called **homoscedasticity**, the errors have constant variance (errors don't increase/decrease as X increases/decreases)
        - **Independence**: observations don't affect each other 

        """,                                    
                            style={'font-family': 'Lora, serif',
                                'font-size': '1.25rem',
                                'color': '#2c3e50',
                                'textAlign': 'left',
                                'max-width': '1000px',
                                'margin': 'auto',
                                'margin-bottom': '20px'
                                                        }, 
                                                mathjax=True),

        html.H2("Now, it's time for you to explore!", 
                                                style={'font-family': 'Montserrat, sans-serif',
                                                        'font-size': '2.0rem',
                                                        'textAlign': 'center',
                                                        'color': 'black',
                                                        'margin-top': '30px'}),

        # html.P(["Have you ever wondered how interconnected the world is? Many things around us seem to have relationships with each other. For example, wingspan and height, caffeine intake and energy levels, study hours and exam scores, and money spent on advertisements and profit earned. However, is there a way we can quantify and visualize these relationships? More importantly, is there a way to empirically show the relationships around us? If only there were a statistical tool we can use to help us do these things…", 
        #                 html.Br(), html.Br(),
        #                 "An answer is ", html.B("linear regression"), ", which helps to quantify the relationship between ", html.B("quantitative (numeric)"), " variables.",
        #                 html.Br(), html.Br(),
        #                 "To explain, think about a farmer selling oranges at a farmers market. Each orange that the farmer sells means more money earned. Let's denote the number of oranges sold as X and the profit earned as Y. In simple linear regression, X is the ", html.B("predictor"), " variable, and Y is the ", html.B("response"), " variable. Simple linear regression aims to quantify and estimate the relationship between X and Y. In our example, let's say that each orange costs $2. Therefore, we know that for each orange that the farmer sells, he makes $2. Or, to put this into statistical language, as X increases by 1, Y increases by 2. This is the essence of linear regression: how much the response Y increases/decreases as the predictor X increases/decreases.",
        #                 html.Br(), html.Br(),
        #                 "Note: one orange for $2 is quite expensive! 🍊"], 
        html.P(["Let's see how this looks in practical terms, by looking at some sample data of how different types of advertising channels impact sales. The red line is called the ", 
                html.B("line of best fit"), ", or the ", html.B("fitted regression line"), ". ", 
                "This line, represented by an equation, represents the ", html.B("model"), " that best fits our data👇"], 
                        style={'font-family': 'Lora, serif',
                                'font-size': '1.25rem',
                                'color': '#2c3e50',
                                'textAlign': 'left',
                                'max-width': '1000px',
                                'margin': 'auto',
                                'margin-bottom': '20px'}),

        # GRAPH 1
        # Dropdown for selecting advertising type
        dcc.Dropdown(
            options=dropdown_options,
            value='TV',  # Default value
            id='dropdown-selection',
            style={'width': '80%', 'margin': 'auto', 'font-family': 'Lora, serif'}
        ),
        html.Div([dcc.Graph(id='graph-content')], style={'width': '45%', 'margin': '0 auto 10px'}),
        # Dynamic comment for the first graph
        html.P(id='dynamic-comment', style={
            'font-family': 'Lora, serif',
            'font-size': '1rem',
            'color': '#7f8c8d',
            'text-align': 'center',
            'max-width': '800px',
            'margin': 'auto',
            'margin-top': '0px',
            'margin-bottom': '40px'
        }),

        # GRAPH 2
        html.P("You can also look at multiple predictors at one time. If you have more than one predictor variable, you must use higher dimensions to visualize your data. 👇", 
                        style={'font-family': 'Lora, serif',
                                'font-size': '1.25rem',
                                'color': '#2c3e50',
                                'textAlign': 'left',
                                'max-width': '1000px',
                                'margin': 'auto',
                                'margin-bottom': '0px'}),

        # dcc.Graph(id="3D_graph"),
        html.Div([dcc.Graph(id='3D_graph')], 
                 style={'width': '45%', 'margin': '0 auto 10px', 'font-family': 'Lora, serif',}),
        html.P("Feel free to filter the data to different values of TV:", style={'padding-left': '10%', 'padding-right': '10%'}),
        html.Div(dcc.RangeSlider(id='range-slider',
                                min=0,  
                                max=300,
                                step=30,  
                                marks={i: str(i) for i in range(0, 301, 30)},
                                value=[0, 300]),
                            style={'width': '80%',
                                'padding-left': '10%',
                                'padding-right': '10%', 
                                'margin': '0 auto 20px'}),
        html.P(id='static-comment', style={
            'font-family': 'Lora, serif',
            'font-size': '1rem',
            'color': '#7f8c8d',
            'text-align': 'center',
            'max-width': '800px',
            'margin': 'auto',
            'margin-top': '0px',
            'margin-bottom': '40px'
        }),

        # GRAPH: PLANE OF BEST FIT
        html.P(["Instead of a line of best fit, what would we have? In 3-dimensional space, we need something that is equivalent to a line in 2-dimensional space. In 3-dimensional space, the equivalent to the line of best fit is called the ",
               html.B("plane of best fit"), "! 👇"], 
                        style={'font-family': 'Lora, serif',
                                'font-size': '1.25rem',
                                'color': '#2c3e50',
                                'textAlign': 'left',
                                'max-width': '1000px',
                                'margin': '0 auto',
                                'margin-bottom': '0px'}),

        html.Div([dcc.Graph(id='plane_of_best_fit')], 
                 style={'width': '45%', 'margin': '0 auto 10px', 'font-family': 'Lora, serif',}),
        html.P(id='plane_comment', style={
            'font-family': 'Lora, serif',
            'font-size': '1rem',
            'color': '#7f8c8d',
            'text-align': 'center',
            'max-width': '800px',
            'margin': 'auto',
            'margin-top': '0px',
            'margin-bottom': '40px'
        }),

        # GRAPH: NORMAL DISTRIBUTION
        html.P(["Let's take a look at the Normality assumption for classical linear regression. A ", 
                html.B("Normal distribution "),
                "has a symmetric, bell-shaped curve, with a ",
                 html.B("mean "),
                 "the center of your data) and ",
                 html.B("standard deviation "), 
                 "(how spread out your data is from the center). 👇"], 
                        style={'font-family': 'Lora, serif',
                                'font-size': '1.25rem',
                                'color': '#2c3e50',
                                'textAlign': 'left',
                                'max-width': '1000px',
                                'margin': 'auto',
                                'margin-bottom': '20px'}),
        html.Div([dcc.Graph(id='normal_distribution_graph')], 
                 style={'width': '45%', 'margin': '0 auto 10px', 'font-family': 'Lora, serif'}),
        html.P(id='static_comment2', style={
            'font-family': 'Lora, serif',
            'font-size': '1rem',
            'color': '#7f8c8d',
            'text-align': 'center',
            'max-width': '800px',
            'margin': 'auto',
            'margin-top': '0px',
            'margin-bottom': '30px'}),
        html.P("Mean:", style={'padding-left': '10%', 'padding-right': '10%'}),
        html.Div(
                dcc.Slider(
                    id="mean", 
                    min=-5, 
                    max=5, 
                    value=0, 
                    marks={-5: '-5', -4:'-4', -3: '-3', -2:'-2', -1:'-1', 0:'0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}),
                                                style={'width': '80%',
                                                        'padding-left': '10%',
                                                        'padding-right': '10%', 
                                                        'margin': '0 auto 20px'}),

        html.P("Standard Deviation:", style={'padding-left': '10%', 'padding-right': '10%'}),
        html.Div(
                dcc.Slider(
                    id="std", 
                    min=1, 
                    max=3, 
                    value=1, 
                    marks={1: '1', 2: '2', 3: '3'}),
                                                 style={'width': '80%',
                                                        'padding-left': '10%',
                                                        'padding-right': '10%', 
                                                        'margin': '0 auto 20px'}),
        html.P("Sample Size:", style={'padding-left': '10%', 'padding-right': '10%'}),
        html.Div(
                dcc.Slider(
                    id="sample_size", 
                    min=0, 
                    max=10000, 
                    step=100, 
                    value=500,
                    marks={0: '0', 1000: '1000', 2000: '2000', 3000: '3000', 4000: '4000', 5000: '5000',
                           6000: '6000', 7000: '7000', 8000: '8000', 9000:'9000', 10000: '10000'}),
                                                style={'width': '80%',
                                                        'padding-left': '10%',
                                                        'padding-right': '10%', 
                                                        'margin': '0 auto 20px'}),

        # GRAPH 4: Make your own scatterplot
        html.P("Now that you've learned the basics of linear regression, it's time for you to have some fun! Try making your own scatterplot! 👇", 
                        style={'font-family': 'Lora, serif',
                                'font-size': '1.25rem',
                                'color': '#2c3e50',
                                'textAlign': 'left',
                                'max-width': '1000px',
                                'margin': 'auto',
                                'margin-bottom': '20px'}),
        
        html.Div([dcc.Graph(id="scatterplot_graph")], 
                 style={'width': '45%', 'margin': '0 auto 10px', 'font-family': 'Lora, serif'}),

        html.P(id='scatterplot_comment', 
            style={
                'font-family': 'Lora, serif',
                'font-size': '1rem',
                'color': '#7f8c8d',
                'text-align': 'center',
                'max-width': '800px',
                'margin': 'auto',
                'margin-top': '0px',
                'margin-bottom': '40px'
            }),
        html.P("Select intercept:", style={'padding-left': '10%', 'padding-right': '10%'}),
        html.Div(
                dcc.Dropdown(
                    id='intercept_dropdown',
                    options=[{'label': str(i), 'value': i} for i in intercepts.values()], 
                    value=0,
                    clearable=False),
                                style={'width': '80%',
                                        'padding-left': '10%',
                                        'padding-right': '10%', 
                                        'margin': '0 auto 20px'}),

        html.P("Select slope:", style={'padding-left': '10%', 'padding-right': '10%'}),
        html.Div(
                dcc.Dropdown(
                    id='slope_dropdown',
                    options=[{'label': str(i), 'value': i} for i in slopes.values()], 
                    value=2,
                    clearable=False),
                                style={'width': '80%',
                                        'padding-left': '10%',
                                        'padding-right': '10%', 
                                        'margin': '0 auto 20px'}),

        html.P("Select sample size:", style={'padding-left': '10%', 'padding-right': '10%'}),
        html.Div(
                dcc.Dropdown(
                    id='n_dropdown',
                    options=[{'label': str(i), 'value': i} for i in n.values()], 
                    value=10,
                    clearable=False),
                                style={'width': '80%',
                                        'padding-left': '10%',
                                        'padding-right': '10%', 
                                        'margin': '0 auto 20px'}),

        html.P("You'll notice that no matter what you set the intercept, slope, and sample size to, the intersection of the dotted red lines will always be somewhere on the regression line. The dotted red lines represent the average of the x values and the average of the y values. These averages always lie on the regression line!", 
                                                    style={ 'font-size': '1.25rem',
                                                            'font-family': 'Lora, serif',
                                                            'color': '#34495e',
                                                            'max-width': '1000px',
                                                            'margin': 'auto',
                                                            'margin-bottom': '20px',
                                                            'line-height': '1.2'}),
                                        
        html.P("Challenge: Can you make the graph represent buying oranges at the farmers market? Remember that the farmer makes $2 for every orange sold! 🍊🧑‍🌾🍊🧑‍🌾🍊🧑‍🌾🍊🧑‍🌾🍊🧑‍🌾", 
                                                    style={ 'font-size': '1.25rem',
                                                            'font-family': 'Lora, serif',
                                                            'color': '#34495e',
                                                            'max-width': '1000px',
                                                            'margin': 'auto',
                                                            'margin-bottom': '20px',
                                                            'line-height': '1.2'}),                                     
]),

        html.P("📈 By visualizing the data, we can see how different explanatory variables have a linear relationship with response variables, following a linear trend, showing how we can fit a linear regression model! 📉", style={
            'font-family': 'Lora, serif',
            'font-size': '1.5rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '40px',
            'line-height': '1.2',
            'font-weight': '700'
        }),

        html.P("However, as datasets grow larger and more complex, these classical models often struggle to capture the intricate, non-linear relationships embedded in the data. This is where Quantum computing comes in, with the potential to revolutionize how we approach regression problems.", 
               style={
                    'font-family': 'Lora, serif',
                    'font-size': '1.25rem',
                    'color': '#34495e',
                    'max-width': '1000px',
                    'margin': 'auto',
                    'margin-top': '5px',
                    'margin-bottom': '20px',
                    'line-height': '1.2'
        }),

        html.P("By harnessing the unique properties of quantum mechanics, quantum models can process information in fundamentally different ways, offering new methods to tackle problems that challenge classical approaches. In this blog post, we will explore the differences between quantum and classical computing in the context of regression, highlighting where quantum computing may provide an edge, but also where classical methods still reign supreme.", 
               style={
                    'font-family': 'Lora, serif',
                    'font-size': '1.25rem',
                    'color': '#34495e',
                    'max-width': '1000px',
                    'margin': 'auto',
                    'margin-top': '5px',
                    'margin-bottom': '20px',
                    'line-height': '1.2'
        }),




        # III. BEYOND REGRESSION: TAKING REGRESSION TO THE NEXT STEP ############################################################################################
        html.H2("III. Beyond Regression: Taking Regression to the Next Step ⏭️", 
                                                style={'font-family': 'Lora, serif',
                                                'font-weight': '700',
                                                'font-size': '2.5rem',
                                                'color': '#2c3e50',
                                                'textAlign': 'center',
                                                'margin-bottom': '20px',
                                                'margin-top': '40px'}),

        # # Add horizontal bar
        # html.Hr(style={
        #     'border': '1px solid #34495e',
        #     'width': '80%',
        #     'margin': '40px auto'
        # }),

        html.P("This concept seems simple enough and quite straightforward on Classical computing with Simple Linear Regression. Then what's the fuss about Quantum computing? ⚛️", style={
            'font-family': 'Montserrat, sans-serif',
            'font-size': '2.0rem',
            'textAlign': 'center',
            'color': 'grey',
            'margin-top': '30px',
            'max-width': '1000px',
            'margin': 'auto'
            # 'margin-bottom': '40px',
            # 'line-height': '1.2',
            # 'font-weight': '500'
        }),

        # # Add horizontal bar
        # html.Hr(style={
        #     'border': '1px solid #34495e',
        #     'width': '80%',
        #     'margin': '40px auto'
        # }),

        html.H2("What is a Quantum Computer?", 
                        style={'font-family': 'Montserrat, sans-serif',
                        'font-size': '2.0rem',
                        'textAlign': 'center',
                        'color': 'black',
                        'margin-top': '30px'}),

        html.P("Alright, let's start by reimagining what a computer can be. Classical computers process information using bits—those simple 0s and 1s that act like tiny light switches, either on or off.  Quantum computers, on the other hand, use qubits (quantum bits), which can be a 0, a 1, or even both 0 and 1 at the same time. This ability to be in multiple states at once is called superposition. Think of a spinning coin: while it's spinning, you can't say if it's heads or tails—it's sort of both until it lands. Quantum computers use this strange property to perform many calculations in parallel, unlike regular computers that focus on one task at a time.", style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '10px',
            'line-height': '1.2'
        }),
        html.P("How does this occur? Thanks to entanglement, a phenomenon where two qubits become so deeply connected that whatever happens to one instantly affects the other—even if they're separated by vast distances. This allows qubits to work together in ways classical bits never could, enabling quantum computers to solve complex problems by sharing and processing information much faster. For certain types of problems, like finding hidden patterns in large datasets, entangled qubits give quantum computers exponential power.", style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '10px',
            'line-height': '1.2'
        }),
        html.Img(src='/assets/Qubit_image.png', style={
        'display': 'block',
        'margin-left': 'auto',
        'margin-right': 'auto',
        'width': '50%'  # Adjust the width as needed
        }),
        html.P("Is it a 1? a 0? or both?", style={
            'font-family': 'Lora, serif',
            'font-size': '1rem',
            'color': '#7f8c8d',
            'text-align': 'center',
            'max-width': '800px',
            'margin': 'auto',
            'margin-top': '0px',
            'margin-bottom': '40px'
        }),
        
        # Insert Image or GIF
        html.Img(src='/assets/entangled.gif', style={
            'width': '50%',
            'display': 'block',
            'margin-left': 'auto',
            'margin-right': 'auto',
            'margin-bottom': '40px'
        }),
        dcc.Markdown("""
                     No matter how far away, the two qubits will always be in opposite states when observed. 

                     **Albert Einstein** referred to this phenomenon as 👻 'Spooky action at a distance' 👻
                     """, style={
            'font-family': 'Lora, serif',
            'font-size': '1rem',
            'color': '#7f8c8d',
            'text-align': 'center',
            'max-width': '800px',
            'margin': 'auto',
            'margin-top': '0px',
            'line-height': '0.7',
            'margin-bottom': '40px'
        }),

        html.H2("The Core Concepts of Quantum Computing ⚛️", 
                    style={'font-family': 'Montserrat, sans-serif',
                    'font-size': '2.0rem',
                    'textAlign': 'center',
                    'color': 'black',
                    'margin-top': '30px'}),

        html.P([
            html.Strong("1. Superposition: "),
            "In classical computing, each bit is either 0 or 1. Quantum computers, on the other hand, leverage superposition, a quantum property that allows qubits to be in a state of 0, 1, or both at the same time. Imagine flipping a coin: while it's spinning, it's neither heads nor tails, but a mix of both. This superposition enables quantum computers to perform many calculations simultaneously."
        ], style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '10px',
            'line-height': '1.2'
        }),

        html.P([
            html.Strong("2. Entanglement: "),
            "Another mind-bending quantum property is entanglement. When qubits are entangled, the state of one qubit instantly influences the state of another, no matter the distance between them. Albert Einstein famously called this “spooky action at a distance.” Entanglement allows quantum computers to share information across qubits in a way that classical bits cannot."
        ], style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '10px',
            'line-height': '1.2'
        }),

        html.P([
            html.Strong("3. Quantum Interference: "),
            "Quantum systems can also use interference to amplify correct solutions and cancel out incorrect ones. This helps quantum computers find the right answers more efficiently, making them well-suited for certain types of optimization and search problems."
        ], style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '10px',
            'line-height': '1.2'
        }),
        html.P("With these unique properties, quantum computers hold the potential to revolutionize fields that require massive parallel computation, such as cryptography, drug discovery, and optimization problems. But, despite their promise, quantum computing is not always the best tool for every job, especially for tasks like linear regression.", style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '30px',
            'line-height': '1.2'
        }),


        html.P("🤔 But wait, what does this all mean for Regression Problems? 🤔", style={
            'font-family': 'Lora, serif',
            'font-size': '1.5rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '40px',
            'line-height': '1.2',
            'font-weight': '700'
        }),

        html.P("Great question! Let's see... ", style={
            'font-family': 'Lora, serif',
            'font-size': '1.5rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '40px',
            'line-height': '1.2',
            'font-weight': '700'
        }),

        # IV. BEYOND REGRESSION: RERESSIONG USING QUANTUM COMPUTERS ############################################################################################
        html.H2("IV. Beyond Regression: Regression using Quantum Computers 💻", style={
            'font-family': 'Lora, serif',
            'font-weight': '700',
            'font-size': '2.5rem',
            'color': '#2c3e50',
            'textAlign': 'center',
            'margin-bottom': '20px',
            'margin-top': '40px'
        }),


            # html.H2("I. Introduction 😃", 
        #                                     style={'font-family': 'Lora, serif',
        #                                     'font-weight': '700',
        #                                     'font-size': '2.5rem',
        #                                     'color': '#2c3e50',
        #                                     'textAlign': 'center',
        #                                     'margin-bottom': '20px',
        #                                     'margin-top': '40px'}),


        html.P("Just like in classical machine learning, the goal of quantum regression is to minimize the error between our predicted values and the actual data. However, in this case, we're using a ⚛️ Variational Quantum Circuit ⚛️ to make the predictions, which introduces several important differences from traditional regression models.", style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '30px',
            'line-height': '1.2'
        }),

        html.P("Let's walk through the process step-by-step of implementing a Quantum Regression Model using Pennylane, a Python package that allows us to build quantum circuits, explaining the quantum elements, and see how the model improves over time as we increase the number of training epochs.", style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '30px',
            'line-height': '1.2'
        }),

        dcc.Markdown(
            '''
            First, install and import the necessary packages using the command:

            ```python
            ! pip install pennylane
            ! pip install pandas
            ! pip install sklearn
            ! pip install plotly

            import pennylane as qml
            from pennylane import numpy as np
            import matplotlib.pyplot as plt
            import pandas as pd
            import sklearn
            from sklearn.preprocessing import StandardScaler
            ```

            We then proceed to import and process our data:

            ```python
            df = pd.read_csv('Advertising.csv')
            TV_ads = df['TV'].values
            Total_sales = df['Sales'].values
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            X = x_scaler.fit_transform(TV_ads.reshape(-1, 1)).flatten()
            Y = y_scaler.fit_transform(Total_sales.reshape(-1, 1)).flatten()
            ```

            ''',
            style={
                'font-family': 'Lora, serif',
                'font-size': '1.2rem',
                'color': '#34495e',
                'max-width': '1000px',
                'margin': 'auto',
                'background-color': '#f4f4f4',
                'padding': '20px',
                'border-radius': '10px',
            }
        ),

        html.P("All good till now, no? Well buckle up! This is where things start to get interesting. 🚀", style={
            'font-family': 'Lora, serif',
            'font-size': '1.5rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-top': '40px',
            'margin-bottom': '40px',
            'line-height': '1.2',
            'font-weight': '700'
        }),

        dcc.Markdown(
            '''
            ```python
            dev = qml.device('default.qubit', wires=2)
            ```

            Instead of using a classical model like LinearRegression from sklearn, we set up a quantum device to simulate a quantum computer. In this case, we're using PennyLane's default simulator to mimic a quantum processor with two qubits (quantum bits).

            ```python
            @qml.qnode(dev)
            def variational_circuit(x, weights):
                # Encoding the input
                qml.RX(x, wires=0)
                qml.RY(x, wires=1)
                
                # Variational layer 1
                qml.RX(weights[0], wires=0)
                qml.RY(weights[1], wires=1)
                qml.CNOT(wires=[0, 1])
                qml.RZ(weights[2], wires=0)
                qml.RX(weights[3], wires=1)
                
                # Variational layer 2
                qml.RY(weights[4], wires=0)
                qml.RX(weights[5], wires=1)
                qml.CNOT(wires=[1, 0])
                qml.RZ(weights[6], wires=1)
                qml.RY(weights[7], wires=0)
                
                return qml.expval(qml.PauliZ(0))
            ```
            ''',
            style={
                'font-family': 'Lora, serif',
                'font-size': '1.2rem',
                'color': '#34495e',
                'max-width': '1000px',
                'margin': 'auto',
                'background-color': '#f4f4f4',  # Optional: background for the code block
                'padding': '20px',  # Optional: padding for the block
                'border-radius': '10px',  # Optional: rounded corners
            }
        ),

        html.P("In classical regression, we might define a linear model like y = mx + b. But here, we're using a quantum circuit to generate our predictions. This function, called variational_circuit, transforms the input x (in this case, the scaled TV ad spending) by applying a series of quantum gates, which manipulate the qubits based on the input and the adjustable weights.", style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '30px',
            'line-height': '1.2'
        }),

        dcc.Markdown(
            '''
            - **Input Encoding**: The values of x are encoded into the quantum states of the qubits using RX and RY gates, which rotate the qubits on the X and Y axes.
            - **Variational Layers**: These are layers of quantum gates controlled by parameters (weights), much like in a neural network. The goal is to adjust these weights to minimize the error in predictions.
            - **Entanglement**: The CNOT gates introduce entanglement between the qubits, allowing them to interact in ways that classical bits can't. 🪄 This is where the quantum magic happens 🪄
            ''',
            style={
                'font-family': 'Lora, serif',
                'font-size': '1.25rem',
                'color': '#34495e',
                'max-width': '1000px',
                'margin': 'auto',
                'line-height': '1.2'
            }
        ),

        html.P("Finally, the circuit outputs an expectation value, which is analogous to a prediction of the target variable (in this case, sales). In other words, we turned our formula into a quantum circuit!", style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '30px',
            'line-height': '1.2'
        }),

        dcc.Markdown(
            '''
            We continue by creating the following function, a simple wrapper around the quantum circuit, taking in an input x and returning the prediction by running the quantum circuit with the current weights.

            ```python
            def variational_regressor(x, weights):
                return variational_circuit(x, weights)
            ```

            Now, just like in classical regression, we use the **Mean Squared Error (MSE)** to minimize the cost function.

            The difference is that instead of using a linear model to make predictions, we're applying it to the quantum circuit itself. This function measures the error between the quantum model's predictions and the actual sales data:

            ```python
            def cost(weights, X, Y):
                predictions = np.array([variational_regressor(x, weights) for x in X])
                return np.mean((predictions - Y) ** 2)
            ```

            We now train the model using Gradient Descent:

            ```python
            weights = np.random.randn(8)
            opt = qml.GradientDescentOptimizer(stepsize=0.01)
            ```
            ''',
            style={
                'font-family': 'Lora, serif',
                'font-size': '1.25rem',
                'color': '#34495e',
                'max-width': '1000px',
                'margin': 'auto',
                'background-color': '#f4f4f4',  # Optional: background for the code block
                'padding': '20px',  # Optional: padding for the block
                'border-radius': '10px',  # Optional: rounded corners
            }
        ),

        html.P("See how some topics overlap? We still use MSE to minimize the cost function, and Gradient Descent to train the model to find the optimal weights!", style={
            'font-family': 'Lora, serif',
            'font-size': '1.5rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-top': '40px',
            'margin-bottom': '40px',
            'line-height': '1.2',
            'font-weight': '700'
        }),

        dcc.Markdown(
            '''
            We now proceed to write the training loop:
            ```python
            batch_size = 20
            epochs = 150
            num_batches = len(X) // batch_size
            epoch_set = {0} | {i for i in range(4, 50, 5)} | {99, 149}
            for epoch in range(epochs):
                for i in range(num_batches):
                    X_batch = X[i*batch_size:(i+1)*batch_size]
                    Y_batch = Y[i*batch_size:(i+1)*batch_size]
                    weights = opt.step(lambda w: cost(w, X_batch, Y_batch), weights)
                if epoch in epoch_set:
                    train_cost = cost(weights, X, Y)
            ```

            In classical linear regression, you typically compute the best-fit line in one step, and boom! You have your model. Quantum regression is a bit more like training a neural network, where the model learns over time by making incremental improvements to its parameters (in this case, the weights of the quantum circuit).

            Finally, we generate predictions and plot the regression line. At the beginning, the line might not fit the data very well, but as we increase the number of epochs (iterations), the model improves. The red regression line will gradually fit the data points more closely as the training progresses.

            ```python
            for i, weights in enumerate(weight_list):
                test_predictions = np.array([variational_regressor(x, weights) for x in X])
                plt.scatter(X, Y, label='TV ads vs Total sales')
                plt.plot(np.linspace(-2, 2, 100), [variational_regressor(x, weights) for x in np.linspace(-2, 2, 100)], label='Regression Line', color='red')
                plt.xlabel('TV_ads (normalized)')
                plt.ylabel('Total_sales (normalized)')
                plt.title(f'Epoch: {epoch_list[i]}')
                plt.legend()
                plt.show()
            ```
            ''',
            style={
                'font-family': 'Lora, serif',
                'font-size': '1.25rem',
                'color': '#34495e',
                'max-width': '1000px',
                'margin': 'auto',
                'background-color': '#f4f4f4',  # Optional: background for the code block
                'padding': '20px',  # Optional: padding for the block
                'border-radius': '10px',  # Optional: rounded corners
            }
        ),

        html.P("Amazing! We got a regression line! 🥳 Now let's observe how the circuit weights predict the sales, and how training epochs make it change over time:", style={
            'font-family': 'Lora, serif',
            'font-size': '1.5rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-top': '40px',
            'margin-bottom': '40px',
            'line-height': '1.2',
            'font-weight': '700'
        }),
        # Quantum Graph with Slider
        html.Div(
            children=[
                html.H2("Quantum Model Predictions", style={
                    'font-family': 'Lora, serif',
                    'font-weight': '700',
                    'font-size': '2rem',
                    'color': '#2c3e50',
                    'textAlign': 'center',
                    'margin-bottom': '20px'
                }),
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
                    dcc.Dropdown(
                        id='dropdown-input',
                        options=[
                            {'label': 'TV', 'value': 'TV'},
                            {'label': 'Radio', 'value': 'Radio'},
                            {'label': 'Newspaper', 'value': 'Newspaper'},
                        ],
                        value='TV',
                        clearable=False,
                    ),
                ], style={'max-width': '800px', 'margin': 'auto', 'margin-bottom': '60px'}),

                html.P("Quantum model results for TV ads vs Sales. See how as epochs increase, the model adjusts its weights to better fit the data.", style={
                    'font-family': 'Lora, serif',
                    'font-size': '1rem',
                    'color': '#7f8c8d',
                    'text-align': 'center',
                    'max-width': '700px',
                    'margin': 'auto',
                    'margin-bottom': '60px'
                }),
                
            ],
            style={'textAlign': 'center'}
        ),
        html.P("As we increase the number of epochs, the model's weights are continually adjusted to better minimize the error between its predictions and the actual data points. At first, the fit is poor, with the line being almost horizontal and missing the data trend. However, as the quantum circuit adjusts its parameters, the regression line begins to better approximate the real relationship between the variables.", style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '30px',
            'line-height': '1.2'
        }),

        html.P("This illustrates the power of quantum regression: just like in neural networks or gradient-optimized classical models, quantum models learn iteratively. What makes quantum models particularly interesting is their ability to capture more complex relationships, as seen in the subtle curvature of the final regression line.", style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '30px',
            'line-height': '1.2'
        }),

        html.P("Why Quantum Computing is Less Suited for Linear Regression 📉", 
                style={'font-family': 'Montserrat, sans-serif',
                        'font-size': '2.0rem',
                        'textAlign': 'center',
                        'color': 'black',
                        'margin-top': '30px'}),

        html.P("While quantum computers can outperform classical ones for certain tasks, linear regression is a problem that classical machines already handle very well. Here are some reasons why quantum computing might not be the best choice for linear regression:", style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '20px',
            'line-height': '1.2'
        }),

        html.P([html.Strong("1. Linear Regression is a Deterministic Task"), ": Linear regression is based on finding a line (or plane) that best fits a dataset by minimizing the distance between the predicted and actual values. The equations that describe this are linear and can be solved deterministically using methods like Ordinary Least Squares (OLS). These are algebraically straightforward tasks that classical computers can handle in a few quick steps."], style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '20px',
            'line-height': '1.2'
        }),

        html.P([html.Strong("Quantum Advantage? "), "Quantum computers shine when solving complex, non-linear problems or tasks that involve searching through large solution spaces. However, linear regression doesn't involve the kind of combinatorial complexity that quantum computers are designed to tackle. The parallelism offered by quantum superposition and entanglement doesn't provide a significant advantage in this case."], style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '20px',
            'line-height': '1.2'
        }),

        html.P([html.Strong("2. Quantum Noise and Instability"), ": Quantum computers are incredibly sensitive to external interference, known as quantum noise. Even the smallest environmental disturbance can cause errors in a quantum system. For simple problems like linear regression, the precision required to calculate regression coefficients may be disrupted by quantum noise."], style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '10px',
            'line-height': '1.2'
        }),

        html.P([html.Strong("Classical computers"), " on the other hand, are deterministic and don't face these noise-related issues, making them much more reliable for tasks that require precise, numerical solutions."], style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '20px',
            'line-height': '1.2'
        }),

        html.P([html.Strong("3. Overhead in Quantum Algorithms"), ": Quantum machine learning algorithms, such as Variational Quantum Circuits (VQCs), are capable of performing tasks like regression. These circuits optimize a set of parameters to reduce the error between the predicted and actual outputs, similar to how classical algorithms like gradient descent work. However, training a quantum model involves a high degree of overhead in terms of iterations, measurements, and parameter updates."], style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '20px',
            'line-height': '1.2'
        }),

        html.P("For linear regression, where the relationship between variables is straightforward, the classical approach reaches a solution quickly with no need for iterative quantum parameter tuning. The extra complexity introduced by quantum circuits doesn't add much benefit for such a simple, linear problem.", style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '20px',
            'line-height': '1.2'
        }),

        html.P([html.Strong("4. Limited Hardware and Qubit Resources"), ": Quantum computers are still in the early stages of development, and the number of qubits available on today's quantum machines is limited. Additionally, these qubits are prone to errors, and performing tasks like error correction consumes even more qubit resources. When it comes to linear regression, which classical computers can handle with ease even on large datasets, the current state of quantum hardware is often overkill."], style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '10px',
            'line-height': '1.2'
        }),

        html.P("Classical systems can easily handle datasets with thousands or millions of records for regression analysis, while quantum computers are still limited in how much data they can process efficiently.", style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '20px',
            'line-height': '1.2'
        }),

        html.P([html.Strong("5. Better Suited for Non-Linear Models"), ": Quantum computers truly shine when they're applied to non-linear problems. In quantum-enhanced models, such as Quantum Support Vector Machines (QSVMs) or Quantum Neural Networks (QNNs), the quantum computer is able to exploit its unique properties to map data into higher-dimensional spaces and uncover hidden patterns in complex datasets."], style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '20px',
            'line-height': '1.2'
        }),

        html.P("However, linear regression is fundamentally about finding linear relationships, which classical methods handle perfectly. Quantum computing might introduce non-linearity where it isn't needed, making it an unnecessary and overly complex approach for this task.", style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '30px',
            'line-height': '1.2'
        }),

        html.P("When Quantum Computing Can Help in Regression Tasks", 
                                        style={'font-family': 'Montserrat, sans-serif',
                                                'font-size': '2.0rem',
                                                'textAlign': 'center',
                                                'color': 'black',
                                                'margin-top': '30px'}),
                
                html.P("Though quantum computers might not excel at simple linear regression, there are certain scenarios where quantum techniques can improve regression models:", style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '20px',
            'line-height': '1.2'
            }),

        html.P([html.Strong("1. Quantum Kernel Methods"), ": Quantum computers can be used to project data into a higher-dimensional feature space using quantum feature maps. In this space, classical linear models can capture complex, non-linear relationships that they would otherwise miss. This approach is useful for tasks like non-linear regression or problems with complicated datasets that exhibit subtle relationships between variables."], style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '20px',
            'line-height': '1.2'
        }),

        html.P([html.Strong("2. Quantum-Assisted Regularization"), ": In high-dimensional datasets, regularization techniques such as Ridge regression or Lasso regression are used to prevent overfitting. Quantum algorithms can assist in optimizing these regularization parameters more efficiently, offering a quantum advantage in certain types of large-scale regression tasks."], style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '10px',
            'line-height': '1.2'
        }),

        html.P([html.Strong("3. Quantum Optimization"), ": Some regression tasks require solving optimization problems, especially when the objective function is non-convex. In these cases, quantum algorithms like Quantum Approximate Optimization Algorithm (QAOA) can be used to find global minima faster than classical optimization techniques."], style={
            'font-family': 'Lora, serif',
            'font-size': '1.25rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '10px',
            'line-height': '1.2'
        }),



# V. Conclusion ################################################################################################################################################
        html.P("V. Conclusion: Classical vs. Quantum for Linear Regression 🧐", style={
                    'font-family': 'Lora, serif',
                    'font-weight': '700',
                    'font-size': '2.5rem',
                    'color': '#2c3e50',
                    'textAlign': 'center',
                    'margin-top': '40px',
                    'margin-bottom': '40px'
                }),


        dcc.Markdown(
            '''
            In summary, while quantum computing is a powerful tool for many types of machine learning and optimization problems, it may not offer significant benefits for simple tasks like linear regression. The deterministic, algebraic nature of linear regression makes it well-suited to classical computers, which can solve these problems efficiently and without the overhead associated with quantum computation.

            That said, quantum computing still holds promise for more complex machine learning tasks, particularly those involving non-linear relationships or high-dimensional data. As quantum hardware improves and algorithms are further developed, we may see more use cases where quantum-enhanced regression models become more practical.

            For now, when it comes to linear regression, classical methods are still the best and most reliable choice.

            This deep dive provides a comprehensive look at the strengths and limitations of quantum computing for linear regression, helping readers understand where quantum technology fits in the broader context of machine learning.

            👩🏽‍⚖️ **Final Verdict** 👩🏽‍⚖️

            Quantum regression is fascinating but still experimental. While its potential is enormous, especially for future large-scale applications, classical models are more efficient and accessible at present. If you're interested in staying on the cutting edge of technology, exploring quantum computing now can be valuable, but for everyday tasks, classical regression remains the more sensible choice.
            ''',
            style={
                'font-family': 'Lora, serif',
                'font-size': '1.25rem',
                'color': '#34495e',
                'max-width': '1000px',
                'margin': 'auto',
                'line-height': '1.2'
            }
        )
    ]
)



########################################################################################################################################################################################################################################################################################################################
# MAKING THE GRAPHS
########################################################################################################################################################################################################################################################################################################################
"""
Hey everyone,

Each section below corresponds to one of the graphs in the previous section, called "CREATING THE LAYOUT".
The X in Output('X', 'figure' ) is what connects to the corresponding graph in "CREATING THE LAYOUT".
Therefore, if you want to quickly access the corresponding graph in "CREATING THE LAYOUT", do Ctrl+F and type what is in place of X.

- Ada 
"""

# GRAPH 1: Callback for updating the FIRST graph and dynamic comment based on dropdown ########################################################################################################################################################################################################################################################################################################################
@app.callback(
    Output('graph-content', 'figure'),
    Output('dynamic-comment', 'children'),
    Input('dropdown-selection', 'value')
)
def update_graph_and_comment(value):
    # Update graph
    fig = px.scatter(advertising_df, x=value, y='Sales', trendline="ols")
    fig.update_traces(marker=dict(color='blue'), selector=dict(mode='markers'))

    fig.update_traces(line=dict(color='red'), selector=dict(mode='lines'))

    fig.update_layout(
        title={'text': f"{value} Impact on Sales", 'x': 0.5, 'xanchor': 'center'},
        hovermode='closest'
    )

    comments = {
        'TV': "This graph shows how TV advertising (independent variable) positively impacts sales (dependent variable), as evidenced by the upward trend in the data points and the fitted regression line. The slope indicates a positive relationship, suggesting that higher TV advertising leads to higher sales.",
        'Radio': "This graph shows how radio advertising positively impacts sales, but with more variability compared to TV advertising, as evidenced by the wider scatter. While both show positive relationships, TV advertising appears to have a stronger and more consistent effect on sales.",
        'Newspaper': "This graph shows how newspaper advertising has a weaker positive impact on sales, as indicated by the flatter slope and more scattered data points compared to TV and radio. The relationship is less consistent, suggesting that newspaper advertising is less effective in driving sales than TV or radio."
    }
    return fig, comments.get(value, "No comment available for this category.")



##################################################################################################################################
# GRAPH 2: Callback for updating the SECOND graph and dynamic comment based on dropdown 
@app.callback(
    Output("3D_graph", "figure"), 
    Output("static-comment", "children"), 
    Input("range-slider", "value"))
def update_bar_chart(slider_range):
    low, high = slider_range
    mask = (advertising_df['TV'] >= low) & (advertising_df['TV'] <= high)

    fig_3D_graph = px.scatter_3d(advertising_df[mask], 
        x='Radio', 
        y='Newspaper', 
        z='Sales',
        hover_data=['TV'])

    fig_3D_graph.update_traces(marker=dict(color='blue'))  
    
    comment = ["Using multiple predictors is called ", html.B("multiple linear regression"), ". Using only one predictor is called ", html.B("simple linear regression"), "."]
    return fig_3D_graph, comment

##################################################################################################################################
# GRAPH 3: PLANE OF BEST FIT
model = LinearRegression() # this fits MLR model
X = advertising_df[['TV', 'Radio']]
y = advertising_df['Sales'] 
model.fit(X, y)
y_pred = model.predict(X)

tv_range = np.linspace(X['TV'].min(), X['TV'].max(), 100)
radio_range = np.linspace(X['Radio'].min(), X['Radio'].max(), 100)
tv_grid, radio_grid = np.meshgrid(tv_range, radio_range)

@app.callback(
    Output('plane_of_best_fit', 'figure'),
    Output("plane_comment", "children"), 
    Input('plane_of_best_fit', 'id')  
)
def update_graph(_):
    X_grid = np.column_stack([tv_grid.ravel(), radio_grid.ravel()])
    y_grid_pred = model.predict(X_grid).reshape(tv_grid.shape)
    fig = go.Figure(data=[go.Surface(z=y_grid_pred, x=tv_grid, y=radio_grid, colorscale='Viridis', opacity=0.7)])
    fig.add_trace(go.Scatter3d(
        x=X['TV'], 
        y=X['Radio'], 
        z=y, 
        mode='markers', 
        marker=dict(size=4, color='blue'), 
        name='Original Data'))
    fig.update_layout(
        scene=dict(
            xaxis_title="TV Advertising Spend ($)",
            yaxis_title="Radio Advertising Spend ($)",
            zaxis_title="Sales ($)"),
        width=800,
        height=600)
    comment = ['This is called the ', html.B("plane of best fit"), '.']

    return fig, comment


##################################################################################################################################
# GRAPH 3: Gaussian Distribution 
@app.callback(
    Output("normal_distribution_graph", "figure"), 
    Output("static_comment2", "children"), 
    Input("mean", "value"), 
    Input("std", "value"),
    Input("sample_size", "value"))
def display_color(mean, std, sample_size):
    data = np.random.normal(mean, std, size=sample_size) 
    fig = px.histogram(data, range_x=[-10, 10], nbins=50, color_discrete_sequence=['blue'])
    fig.data[0].showlegend = False
    comment = ["Visualizing the ", html.B("Normal distribution"), ", one of the assumptions for regression. Try changing the mean, standard deviation, and/or sample size. How does the distribution change?"]
    return fig, comment


##################################################################################################################################
# GRAPH 4: Make your own scatterplot

@app.callback(
    Output("scatterplot_graph", "figure"), 
    Output("scatterplot_comment", "children"), 
    Input("intercept_dropdown", "value"), 
    Input("slope_dropdown", "value"),
    Input("n_dropdown", "value"),
    prevent_initial_callback=True
)
def display_scatter(intercept_dropdown, slope_dropdown, n_dropdown):
    if intercept_dropdown is None:
        intercept_dropdown = 0
    x_data, y_data, x_mean, y_mean = generate_data(intercept_dropdown, slope_dropdown, n_dropdown)
    fig = px.scatter(x=x_data, y=y_data, trendline='ols', color_discrete_sequence=["blue"])
    fig.update_traces(line=dict(color="black"))
    fig.add_vline(x=x_mean, line_dash="dash", line_color="red")
    fig.add_hline(y=y_mean, line_dash="dash", line_color="red")
    fig.data[0].showlegend = False

    comment = 'Make a scatterplot! Change the intercept, slope, and/or sample size, and notice how the plot behaves. Do you notice how the regression line changes? Do you notice anything about the dotted red lines?'
    return fig, comment

##################################################################################################################################
# QUANTUM GRAPH: Callback for updating the quantum model graph based on the slider 
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



########################################################################################################################################################################################################################################################################################################################
# RUN THE APP
########################################################################################################################################################################################################################################################################################################################
if __name__ == '__main__':
    app.run(debug=True)