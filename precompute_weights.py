import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('Advertising.csv')
for ad_type in ['TV', 'Radio', 'Newspaper']:
    ads = df[ad_type].values
    Total_sales = df['Sales'].values

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X = x_scaler.fit_transform(ads.reshape(-1, 1)).flatten()
    Y = y_scaler.fit_transform(Total_sales.reshape(-1, 1)).flatten()

    # define quantum device
    dev = qml.device('default.qubit', wires=2)

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

    '''variational_circuit(x, weights), which is the actual quantum circuit (discussed earlier). 
    The circuit performs calculations based on the input x and the weights and returns a prediction.'''
    def variational_regressor(x, weights):
        return variational_circuit(x, weights)

    '''Let's use MSE to minimize the cost function: '''
    def cost(weights, X, Y):
        predictions = np.array([variational_regressor(x, weights) for x in X])
        return np.mean((predictions - Y) ** 2)

    '''Initialize the weights of the Quantum Circuit randomly'''
    np.random.seed(42)
    weights = np.random.randn(8)
    ''' Using Gradient Descent we chose to minimize the cost function by adjusting the weights of the circuit. '''
    opt = qml.GradientDescentOptimizer(stepsize=0.01)

    batch_size = 20
    epochs = 150
    weight_list = []
    num_batches = len(X) // batch_size
    epoch_set = {0} | {i for i in range(9, 50, 10)} | {99, 149}
    for epoch in range(epochs):
        for i in range(num_batches):
            X_batch = X[i*batch_size:(i+1)*batch_size]
            Y_batch = Y[i*batch_size:(i+1)*batch_size]
            weights = opt.step(lambda w: cost(w, X_batch, Y_batch), weights)
        if epoch in epoch_set:
            train_cost = cost(weights, X, Y)
            weight_list.append(weights)
            print(f"Epoch {epoch + 1}: Train cost = {train_cost}")
    np.save(f'{ad_type}-weights.npy', weight_list)