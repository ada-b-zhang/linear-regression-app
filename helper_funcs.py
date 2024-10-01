import pennylane as qml

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

def variational_regressor(x, weights):
    return variational_circuit(x, weights)