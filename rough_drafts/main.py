# Import all packages
from dash import Dash, html, dcc, Output, Input
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from helper_funcs import variational_regressor
from sklearn.preprocessing import StandardScaler

# Open the dataset
advertising_df = pd.read_csv('advertising.csv')
advertising_df = advertising_df.drop('Unnamed: 0', axis=1)
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

# Initialize Dash app
app = Dash(
    suppress_callback_exceptions=True,
    external_stylesheets=[
        'https://fonts.googleapis.com/css2?family=Lora:wght@400;700&family=Montserrat:wght@300&display=swap',
        'https://fonts.googleapis.com/css2?family=Sixtyfour+Convergence&display=swap'
    ]
)

# Create dropdown options by mapping column names to custom labels
column_labels = {
    'TV': 'TV Advertising',
    'Radio': 'Radio Advertising',
    'Newspaper': 'Newspaper Advertising'
}
dropdown_options = [{'label': column_labels[col], 'value': col} for col in advertising_df.columns[:-1]]

# Create layout
app.layout = html.Div(
    children=[
        # Title and Subtitle
        html.Header(
            children=[
                html.H1("Quantum vs Classical Computing", style={
                    'font-family': 'Sixtyfour Convergence, serif',
                    'font-weight': '800',
                    'font-size': '3.6rem',
                    'color': '#2c3e50',
                    'textAlign': 'center',
                    'margin-bottom': '20px'
                }),
                html.P("Explaining the difference between Quantum and Classical Computing for Regression Problems", style={
                    'font-family': 'Montserrat, sans-serif',
                    'font-size': '1.2rem',
                    'textAlign': 'center',
                    'color': '#7f8c8d',
                    'margin-top': '30px',
                    'margin-bottom': '60px'
                })
            ]
        ),

        # First paragraphs
        html.Div(
            children=[
                html.P("In the world of data science, Linear regression is one of the most basic techniques used to discover relationships between variables, and make predictions. Traditionally, classical computing has been the backbone of this process, with linear and non-linear regression models being widely employed.", style={
                    'font-family': 'Lora, serif',
                    'font-size': '1.5rem',
                    'color': '#34495e',
                    'max-width': '1000px',
                    'margin': 'auto',
                    'margin-top': '5px',
                    'margin-bottom': '20px',
                    'line-height': '1.2'
                }),

                html.P("However, as datasets grow larger and more complex, these classical models often struggle to capture the intricate, non-linear relationships embedded in the data. This is where Quantum computing comes in, with the potential to revolutionize how we approach regression problems.", style={
                    'font-family': 'Lora, serif',
                    'font-size': '1.5rem',
                    'color': '#34495e',
                    'max-width': '1000px',
                    'margin': 'auto',
                    'margin-top': '5px',
                    'margin-bottom': '20px',
                    'line-height': '1.2'
                }),

                html.P("By harnessing the unique properties of quantum mechanics, quantum models can process information in fundamentally different ways, offering new methods to tackle problems that challenge classical approaches. In this blog post, we will explore the differences between quantum and classical computing in the context of regression, highlighting where quantum computing may provide an edge, but also where classical methods still reign supreme.", style={
                    'font-family': 'Lora, serif',
                    'font-size': '1.5rem',
                    'color': '#34495e',
                    'max-width': '1000px',
                    'margin': 'auto',
                    'margin-top': '5px',
                    'margin-bottom': '20px',
                    'line-height': '1.2'
                }),
            ]
        ),

        # Add horizontal bar
        html.Hr(style={
            'border': '1px solid #34495e',
            'width': '80%',
            'margin': '40px auto'
        }),

        # Heading for Linear Regression Explanation
        html.H2("How does Classical Linear Regression work?", style={
            'font-family': 'Lora, serif',
            'font-weight': '700',
            'font-size': '2.5rem',
            'color': '#2c3e50',
            'textAlign': 'left',
            'margin-left': '200px',
            'margin-bottom': '20px',
            'margin-top': '40px'
        }),

        # Normal explanation of Linear Regression
        html.P("Linear regression is one of the simplest and most widely used predictive modeling techniques. It models the relationship between a dependent variable and one or more independent variables. The objective is to find a linear equation that best fits the data points. This technique assumes that the relationship between the dependent and independent variables is linear, this is, a straight line can be drawn through the data points to find a relationship.", style={
            'font-family': 'Lora, serif',
            'font-size': '1.5rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '20px',
            'line-height': '1.2'
        }),

        # Mathematical explanation of Linear Regression
        html.P("In mathematical terms, the linear regression model can be represented as:", style={
            'font-family': 'Lora, serif',
            'font-size': '1.5rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '10px',
            'line-height': '1.2'
        }),
        dcc.Markdown("""
        $$
        y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + \dots + \\beta_n x_n + \epsilon
        $$
        """, style={
            'font-family': 'Lora, serif',
            'font-size': '1.5rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '20px',
            'line-height': '1.2',
            'textAlign': 'center'
        }, mathjax=True),

        html.P("Where:", style={
            'font-family': 'Lora, serif',
            'font-size': '1.5rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '10px',
            'line-height': '1.2'
        }),
        dcc.Markdown("""
        - $y$ is the dependent variable (the outcome we are trying to predict).
        - $x_1, x_2, ..., x_n$ are the independent variables (the predictors).
        - $\\beta_0$ is the intercept (the value of $y$ when all $x_i$ are zero).
        - $\\beta_1, \\beta_2, ..., \\beta_n$ are the coefficients (the impact of each $x_i$ on $y$).
        - $\\epsilon$ is the error term (the difference between the observed and predicted values).
        """, style={
            'font-family': 'Lora, serif',
            'font-size': '1.5rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'line-height': '1.5'
        }, mathjax=True),

        html.P("👇 Let's see how this looks in practical terms, by looking at some sample data of how different types of advertising channels impact sales 👇", style={
            'font-family': 'Lora, serif',
            'font-size': '1.5rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '40px',
            'line-height': '1.2',
            'font-weight': '700'
        }),

        # Dropdown for selecting advertising type
        dcc.Dropdown(
            options=dropdown_options,
            value='TV',  # Default value
            id='dropdown-selection',
            style={'width': '80%', 'margin': 'auto', 'font-family': 'Lora, serif'}
        ),
        # First graph
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

        html.P("By visualizing the data, we can see how different types of advertising channels have a linear relationship with sales, following a linear trend, showing how we can fit a Linear Regression model.", style={
            'font-family': 'Lora, serif',
            'font-size': '1.5rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '10px',
            'line-height': '1.2'
        }),

        html.P("This concept seems simple enough and quite straightforward on Classical computing with Simple Linear Regression. Then what's the fuss about Quantum computing? ⚛️", style={
            'font-family': 'Lora, serif',
            'font-size': '1.5rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '40px',
            'line-height': '1.2',
            'font-weight': '700'
        }),

        # Add horizontal bar
        html.Hr(style={
            'border': '1px solid #34495e',
            'width': '80%',
            'margin': '40px auto'
        }),

        html.H2("What is a Quantum Computer?", style={
            'font-family': 'Lora, serif',
            'font-weight': '700',
            'font-size': '2.5rem',
            'color': '#2c3e50',
            'textAlign': 'left',
            'margin-left': '200px',
            'margin-bottom': '20px',
            'margin-top': '40px'
        }),

        html.P("Alright, let’s start by reimagining what a computer can be. Classical computers process information using bits: those simple 0s and 1s that act like tiny light switches, either on or off. Quantum computers, on the other hand, take that idea and throw it into an entirely new dimension. Literally. By using qubits", style={
            'font-family': 'Lora, serif',
            'font-size': '1.5rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '10px',
            'line-height': '1.2'
        }),
        html.P("A qubit (quantum bit) can be a 0, a 1 (just like a normal bit) or, here’s the mind-bending part, both 0 and 1 at the same time! This ability to be in multiple states at once is called superposition. Think of a spinning coin: while it’s spinning, you can’t say if it’s heads or tails—it’s sort of both until it lands. Quantum computers harness this strange property to perform many calculations in parallel, unlike your regular laptop that can only focus on one task at a time.", style={
            'font-family': 'Lora, serif',
            'font-size': '1.5rem',
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
        html.P("How does this occur? Because of a property called entanglement. Imagine two of these qubits so deeply connected that whatever happens to one instantly affects the other—even if they’re separated by a million miles! This is called quantum entanglement, and it’s one of the wildest, most mind-boggling concepts in quantum physics. It’s like having two magical coin: flip one, and no matter where the other one is, it will instantly show the opposite result.", style={
            'font-family': 'Lora, serif',
            'font-size': '1.5rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '10px',
            'line-height': '1.2'
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

        html.P("For quantum computers, this means that qubits can work together in ways classical bits never could. This deep connection allows quantum computers to solve complex problems by sharing and processing information much faster, like two brains that can think as one. When classical computers hit a wall trying to compute certain things (like finding hidden patterns in complex datasets), entangled qubits allow quantum computers to tackle these tasks in parallel, making them exponentially more powerful for certain types of problems.", style={
            'font-family': 'Lora, serif',
            'font-size': '1.5rem',
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
        # Add horizontal bar
        html.Hr(style={
            'border': '1px solid #34495e',
            'width': '80%',
            'margin': '40px auto'
        }),

        html.H2("Regression using Quantum Computers", style={
            'font-family': 'Lora, serif',
            'font-weight': '700',
            'font-size': '2.5rem',
            'color': '#2c3e50',
            'textAlign': 'left',
            'margin-left': '200px',
            'margin-bottom': '20px',
            'margin-top': '40px'
        }),

        html.P("Just like in classical machine learning, the goal of quantum regression is to minimize the error between our predicted values and the actual data. However, in this case, we’re using a ⚛️ Variational Quantum Circuit ⚛️ to make the predictions, which introduces several important differences from traditional regression models.", style={
            'font-family': 'Lora, serif',
            'font-size': '1.5rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '30px',
            'line-height': '1.2'
        }),

        html.P("Let’s walk through the process step-by-step of implementing a Quantum Regression Model using Pennylane, a Python package that allows us to build quantum circuits, explaining the quantum elements, and see how the model improves over time as we increase the number of training epochs.", style={
            'font-family': 'Lora, serif',
            'font-size': '1.5rem',
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

            Instead of using a classical model like LinearRegression from sklearn, we set up a quantum device to simulate a quantum computer. In this case, we’re using PennyLane’s default simulator to mimic a quantum processor with two qubits (quantum bits).

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

        html.P("In classical regression, we might define a linear model like y = mx + b. But here, we’re using a quantum circuit to generate our predictions. This function, called variational_circuit, transforms the input x (in this case, the scaled TV ad spending) by applying a series of quantum gates, which manipulate the qubits based on the input and the adjustable weights.", style={
            'font-family': 'Lora, serif',
            'font-size': '1.5rem',
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
            - **Entanglement**: The CNOT gates introduce entanglement between the qubits, allowing them to interact in ways that classical bits can’t. 🪄 This is where the quantum magic happens 🪄
            ''',
            style={
                'font-family': 'Lora, serif',
                'font-size': '1.5rem',
                'color': '#34495e',
                'max-width': '1000px',
                'margin': 'auto',
                'line-height': '1.2'
            }
        ),

        html.P("Finally, the circuit outputs an expectation value, which is analogous to a prediction of the target variable (in this case, sales). In other words, we turned our formula into a quantum circuit!", style={
            'font-family': 'Lora, serif',
            'font-size': '1.5rem',
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

            The difference is that instead of using a linear model to make predictions, we’re applying it to the quantum circuit itself. This function measures the error between the quantum model’s predictions and the actual sales data:

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
                'font-size': '1.2rem',
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
                'font-size': '1.2rem',
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
                    )
                ], style={'max-width': '800px', 'margin': 'auto', 'margin-bottom': '60px'}),
                html.P("Quantum model results for TV ads vs Sales. See how as epochs increase, the model adjusts its weights to better fit the data.", style={
                    'font-family': 'Lora, serif',
                    'font-size': '1rem',
                    'color': '#7f8c8d',
                    'text-align': 'center',
                    'max-width': '700px',
                    'margin': 'auto',
                    'margin-bottom': '60px'
                })
            ],
            style={'textAlign': 'center'}
        ),
        html.P("As we increase the number of epochs, the model’s weights are continually adjusted to better minimize the error between its predictions and the actual data points. At first, the fit is poor, with the line being almost horizontal and missing the data trend. However, as the quantum circuit adjusts its parameters, the regression line begins to better approximate the real relationship between the variables.", style={
            'font-family': 'Lora, serif',
            'font-size': '1.5rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '30px',
            'line-height': '1.2'
        }),

        html.P("This illustrates the power of quantum regression: just like in neural networks or gradient-optimized classical models, quantum models learn iteratively. What makes quantum models particularly interesting is their ability to capture more complex relationships, as seen in the subtle curvature of the final regression line.", style={
            'font-family': 'Lora, serif',
            'font-size': '1.5rem',
            'color': '#34495e',
            'max-width': '1000px',
            'margin': 'auto',
            'margin-bottom': '30px',
            'line-height': '1.2'
        }),

        html.P("🤔 So... what's the difference? Isn't it already good enough? Why go through all this? 🤔", style={
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

        html.Hr(style={
            'border': '1px solid #34495e',
            'width': '80%',
            'margin': '40px auto'
        }),

        html.H2("Differences... and Conclusions", style={
            'font-family': 'Lora, serif',
            'font-weight': '700',
            'font-size': '2.5rem',
            'color': '#2c3e50',
            'textAlign': 'left',
            'margin-left': '200px',
            'margin-bottom': '20px',
            'margin-top': '40px'
        }),
        dcc.Markdown(
            '''
            In classical regression, you would fit a model like y = mx + b using a deterministic, algebraic approach. With quantum regression, instead of a simple formula, we’re using a quantum circuit to make predictions. The quantum circuit is much more flexible because it can model non-linear relationships inherently due to the complexity of quantum states.

            - **Flexibility**: The quantum circuit allows us to capture complex, non-linear patterns that linear regression can’t. This can be especially useful when the data isn’t well-suited to a straight line.
            - **Training Process**: n classical linear regression, finding the best-fit line is typically a one-step process. In quantum regression, we need to optimize the parameters (weights) iteratively, similar to training a neural network.
            - **Performance over time**: As we increase the epochs, the model gradually learns a better fit for the data. Early on, the regression line may be far off, but by the end of training, it should closely follow the data points, reflecting a better understanding of the relationship between TV ads and sales.

            As you can see, quantum regression introduces a whole new way of thinking about predictive modeling. While it’s still an emerging field, the potential for quantum computers to uncover hidden patterns in complex data could open up exciting new possibilities, particularly for non-linear relationships where classical methods struggle. Over time, as the model trains and we increase the epochs, the quantum model becomes better at predicting sales based on TV ad spending—similar to how traditional models improve, but with the unique power of quantum mechanics behind it.
            ''',
            style={
                'font-family': 'Lora, serif',
                'font-size': '1.5rem',
                'color': '#34495e',
                'max-width': '1000px',
                'margin': 'auto',
                'line-height': '1.2'
            }
        ),
        html.P("🤔 Is it worth it though? 🤔", style={
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
            👍 **Potential Benefits** 👍

            Quantum regression excels at capturing non-linear relationships in data, which classical linear models often miss. The expressive power of quantum circuits, utilizing entanglement and superposition, allows them to handle complex patterns naturally without the need for manually engineered features. This could give quantum models an edge in problems where traditional models struggle to fit non-linear dynamics. As quantum hardware improves, this potential will only grow, especially for tackling large, complex datasets.

            👎 **Current Drawbacks** 👎

            Despite its promise, quantum regression is currently slow and computationally expensive. Training requires many epochs to converge, and quantum simulators or hardware are still limited, making quantum models impractical for most users today. Classical models, on the other hand, are fast, accessible, and well-optimized for most datasets. Additionally, quantum models are harder to interpret, meaning the tradeoff for flexibility is a loss in transparency compared to classical methods like linear regression.

            ⚛️ **Should you use Quantum Computers for regression?** ⚛️

            For now, classical methods are more practical for most real-world regression problems, especially if your data is small or the relationships are simple. Quantum regression might be worth exploring if you’re dealing with highly complex, non-linear data and have access to the necessary computational resources. However, for most users, classical models like linear or polynomial regression will continue to be the go-to solution.

            👩🏽‍⚖️ **Final Verdict** 👩🏽‍⚖️

            Quantum regression is fascinating but still experimental. While its potential is enormous, especially for future large-scale applications, classical models are more efficient and accessible at present. If you’re interested in staying on the cutting edge of technology, exploring quantum computing now can be valuable, but for everyday tasks, classical regression remains the more sensible choice.
            ''',
            style={
                'font-family': 'Lora, serif',
                'font-size': '1.5rem',
                'color': '#34495e',
                'max-width': '1000px',
                'margin': 'auto',
                'line-height': '1.2'
            }
        )
    ]
)

# Callback for updating the main graph and dynamic comment based on dropdown
@app.callback(
    Output('graph-content', 'figure'),
    Output('dynamic-comment', 'children'),
    Input('dropdown-selection', 'value')
)
def update_graph_and_comment(value):
    # Update graph
    fig = px.scatter(advertising_df, x=value, y='Sales', trendline="ols")
    fig.update_layout(
        title={'text': f"{value} Impact on Sales", 'x': 0.5, 'xanchor': 'center'},
        hovermode='closest'
    )

    # Update comment
    comments = {
        'TV': "This graph shows how TV advertising (independent variable) positively impacts sales (dependent variable), as evidenced by the upward trend in the data points and the fitted regression line. The slope indicates a positive relationship, suggesting that higher TV advertising leads to higher sales.",
        'Radio': "This graph shows how radio advertising positively impacts sales, but with more variability compared to TV advertising, as evidenced by the wider scatter. While both show positive relationships, TV advertising appears to have a stronger and more consistent effect on sales.",
        'Newspaper': "This graph shows how newspaper advertising has a weaker positive impact on sales, as indicated by the flatter slope and more scattered data points compared to TV and radio. The relationship is less consistent, suggesting that newspaper advertising is less effective in driving sales than TV or radio."
    }
    return fig, comments.get(value, "No comment available for this category.")

# Callback for updating the quantum model graph based on the slider
@app.callback(
    Output('graph-output', 'figure'),
    Input('weight-slider', 'value')
)
def update_quantum_graph(epoch_value):
    index = epoch_values.index(epoch_value)
    weight_list = weight_lists[index]
    original_X = x_scaler.inverse_transform(np.linspace(-2, 2, 100).reshape(-1, 1)).flatten()

    predicted_Y = [variational_regressor(x, weight_list) for x in np.linspace(-2, 2, 100)]
    unnormalized_Y = y_scaler.inverse_transform(np.array(predicted_Y).reshape(-1, 1)).flatten()

    fig = px.scatter(x=x_scaler.inverse_transform(X.reshape(-1, 1)).flatten(),
                     y=y_scaler.inverse_transform(Y.reshape(-1, 1)).flatten())
    fig.add_trace(go.Scatter(x=original_X, y=unnormalized_Y, mode='lines', line=dict(color='red'), name='Fitted Line'))
    fig.update_layout(
        title={'text': "Quantum TV Predictions Per Epoch", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='TV Advertisement Spending (in thousands)',
        yaxis_title='Total Sales (in millions)'
    )
    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)