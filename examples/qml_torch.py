# This code is standalone.
#
# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.



import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import copy

import torch
from torch.nn import Linear, CrossEntropyLoss, MSELoss, BCELoss, Softmax
from torch.optim import Adam

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient


from gspsa_gradients.qiskit_gradient import GSPSAEstimatorGradient



def train(model, train_data, val_data, optimizer, f_loss, num_epochs, method, patience):
    best_acc = 0.0
    best_model = copy.deepcopy(model)
    earlyStopping = 0
    for epoch in range(num_epochs):
        model.train()  # set model to training mode
        optimizer.zero_grad()  # Initialize/clear gradients
        out = model(train_data[0])
        out = torch.nn.Sigmoid()(out)
        loss = f_loss(out, train_data[1])  # Evaluate loss function
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights using Adam optimizer
        
        # Compute classification accuracy
        model.eval()
        out = model(val_data[0])
        predictions = torch.argmax(out, dim=1)
        predictions = predictions.detach().cpu().numpy()
        targets = torch.argmax(val_data[1], dim=1)
        correct = np.sum((predictions == targets.detach().cpu().numpy()))
        accuracy = correct / len(val_data[0]) * 100

        if accuracy > best_acc:
            earlyStopping = 0
            best_acc = accuracy
            best_model = copy.deepcopy(model)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy {method}: {accuracy:.2f}%")
        earlyStopping += 1
        if earlyStopping >= patience:
            break
    return best_model

def plot_prediction(model, X_, y, file_name):
    model.eval()
    out = model(X_) 
    out = torch.nn.Sigmoid()(out)
    predictions = torch.argmax(out, dim=1)
    predictions = predictions.detach().cpu().numpy()
    y = torch.argmax(y, dim=1).detach().cpu().numpy()
    correct = np.sum((predictions == y))
    accuracy = correct / len(y) * 100
    fig, ax = plt.subplots()
    for x, y_target, y_p in zip(X_, y, predictions):
        if y_target == 1:
            ax.plot(x[0], x[1], "bo")
        else:
            ax.plot(x[0], x[1], "go")
        if y_target != y_p:
            ax.scatter(x[0], x[1], s=200, facecolors="none", edgecolors="r", linewidths=2)
    ax.set_title(f"Test accuracy : {accuracy:.2f}%")
    fig.savefig(file_name)
    plt.close()

# Set seed for random generators
algorithm_globals.random_seed = 42

# Generate random dataset
num_inputs = 2
num_samples = 400
num_classes = 2
num_epochs = 50  # Set the number of training epochs, This will be used as the steps for the gspsa
patience = 10



# Generate the dataset
X, y = make_blobs(n_samples=num_samples, centers=num_classes, n_features=num_inputs, random_state=42, cluster_std=3)



# Split the dataset into train, test, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)



# Normalize the datasets between 0 and 1
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)


for x, y_target in zip(X_test_norm, y_test):
    if y_target == 1:
        plt.plot(x[0], x[1], "bo")
    else:
        plt.plot(x[0], x[1], "go")
plt.savefig("examples/qml_torch_data.png")

# One-hot encode the class labels
y_train_ohe = np.eye(num_classes)[y_train]
y_val_ohe = np.eye(num_classes)[y_val]
y_test_ohe = np.eye(num_classes)[y_test]

# Convert all data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_norm, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train_ohe, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_ohe, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_ohe, dtype=torch.float32)


# Set up a circuit
n_layers = 5
qc = QuantumCircuit(num_classes)
inputParams = ParameterVector("x", length=num_classes)
circuitParams = ParameterVector("psi", length=2*n_layers*num_classes)
data_counter = 0
# Construct the variational layers
for i in range(n_layers):
    for j in range(num_classes):
        if data_counter != num_inputs:
           qc.rx(inputParams[data_counter], j)
           data_counter += 1 
        qc.ry(circuitParams[2*i*num_inputs + j], j)
        qc.rz(circuitParams[(2*i+1)*num_inputs + j], j)
    
    for j in range(num_classes-1):
        qc.cz(j, (j+1))
    qc.barrier()
qc.draw(output='mpl', filename='examples/qml_torch_circuit.png')
plt.close()

observables = [] 
for i in range(num_classes):
    Pauli_string = "I"*num_classes
    Pauli_string = Pauli_string[:i] + 'Z' + Pauli_string[i+1:]
    observables.append(SparsePauliOp(Pauli(Pauli_string)))

# Set up PyTorch module
initial_weights = 0.1 * (2 * algorithm_globals.random.random(2*n_layers*num_inputs) - 1)


# Setup QNN - SPSA
gradient = GSPSAEstimatorGradient(total_steps = num_epochs, estimator=Estimator(), num_observables = len(observables), 
    tau=0.5, spsa_epsilon=0.01, damping_coeff=0.5)
qnn1 = EstimatorQNN(
    circuit=qc, input_params=inputParams, weight_params=circuitParams, observables=observables, gradient=gradient, input_gradients=False)

# Set up PyTorch module
model = TorchConnector(qnn1, initial_weights=initial_weights)
print("Initial weights: ", initial_weights)

# Define optimizer and loss
optimizer = Adam(model.parameters(), lr=0.1)  # Use Adam optimizer with default learning rate
f_loss = BCELoss()

# Start training
model = train(model=model, train_data=(X_train_tensor, y_train_tensor), val_data = (X_val_tensor, y_val_tensor), 
     optimizer=optimizer, f_loss=f_loss, num_epochs=num_epochs, method='GSPSA', patience=patience)
plot_prediction(model=model, X_= X_test_tensor, y=y_test_tensor, file_name= "examples/qml_torch_predictions_GSPSA.png")
