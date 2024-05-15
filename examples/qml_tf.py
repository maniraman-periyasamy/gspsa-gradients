import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import copy

import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy

from gspsa_gradients.tfq_gradient import GSPSAGradient


tf.config.run_functions_eagerly(True)

def train(model, train_data, val_data, optimizer, loss_fn, num_epochs, method, patience):
    best_acc = 0.0
    best_model = copy.deepcopy(model)
    earlyStopping = 0
    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            logits = model(train_data[0], training=True)
            loss = loss_fn(train_data[1], logits)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Compute classification accuracy
        logits = model(val_data[0], training=False)
        predictions = tf.argmax(logits, axis=1)
        targets = tf.argmax(val_data[1], axis=1)
        correct = tf.reduce_sum(tf.cast(tf.equal(predictions, targets), tf.int32))
        accuracy = correct / len(val_data[0]) * 100

        if accuracy > best_acc:
            earlyStopping = 0
            best_acc = accuracy
            best_model = model

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}, Accuracy {method}: {accuracy:.2f}%")
        earlyStopping += 1
        if earlyStopping >= patience:
            break
    return best_model

def plot_prediction(model, X_, y, file_name):
    logits = model(X_, training=False)
    predictions = tf.argmax(logits, axis=1)
    y = tf.argmax(y, axis=1)
    correct = tf.reduce_sum(tf.cast(tf.equal(predictions, y), tf.int32))
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


class quantum_layer(tf.keras.layers.Layer):
    

    def __init__(self, circuit, theta_symbols, input_symbols, observables, diff):
        super(quantum_layer, self).__init__()
        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=1)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True, name="thetas"
        )

        # Define explicit symbol order.
        symbols = [str(symb) for symb in theta_symbols + input_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])

        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables, differentiator=diff, repetitions=1024) # make repitions to 0 for exact expectation value estimation         

    def call(self, inputs):
        # inputs[0] = encoding data for the state.
        batch_dim = tf.shape(inputs)[0]
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1])
        joined_vars = tf.concat([tiled_up_thetas, inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        return self.computation_layer([tiled_up_circuits, joined_vars])


# Set seed for random generators
np.random.seed(42)

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
plt.savefig("examples/qml_tfq_data.png")

# One-hot encode the class labels
y_train_ohe = tf.one_hot(y_train, depth=num_classes)
y_val_ohe = tf.one_hot(y_val, depth=num_classes)
y_test_ohe = tf.one_hot(y_test, depth=num_classes)

# Convert all data to TensorFlow tensors
X_train_tensor = tf.convert_to_tensor(X_train_norm, dtype=tf.float32)
X_val_tensor = tf.convert_to_tensor(X_val_norm, dtype=tf.float32)
X_test_tensor = tf.convert_to_tensor(X_test_norm, dtype=tf.float32)

y_train_tensor = tf.convert_to_tensor(y_train_ohe, dtype=tf.float32)
y_val_tensor = tf.convert_to_tensor(y_val_ohe, dtype=tf.float32)
y_test_tensor = tf.convert_to_tensor(y_test_ohe, dtype=tf.float32)

# Set up a circuit
n_layers = 5
qubits = cirq.GridQubit.rect(1, num_classes) #tfq.create_circuit(num_classes, name='qubits')
circuitParams = sympy.symbols(f'theta(0:{2*n_layers*num_classes})')
inputParams = sympy.symbols(f'x(0:{num_classes})')
circuit = cirq.Circuit()
# Construct the variational layers
data_counter = 0
for i in range(n_layers):
    for j in range(num_classes):
        if data_counter < num_inputs:
            circuit.append(cirq.rx(inputParams[j]).on(qubits[j]))
            data_counter += 1
        circuit.append(cirq.ry(circuitParams[2*i*num_classes + j]).on(qubits[j]))
        circuit.append(cirq.rz(circuitParams[(2*i+1)*num_classes + j]).on(qubits[j]))

    for j in range(num_classes-1):
        circuit.append(cirq.CZ.on(qubits[j], qubits[j+1]))


initial_weights = tf.Variable(tf.random.uniform([2*n_layers*num_classes], minval=0, maxval=1))

# Set up the quantum circuit
obs  = [cirq.PauliString(cirq.Z(q)) for q in qubits]


# Set up the GSPSA differentiator

ql = quantum_layer(circuit=circuit, theta_symbols=circuitParams, input_symbols=inputParams, 
    observables=obs, diff=GSPSAGradient(total_steps=num_epochs, input_dims=num_inputs, tau=0.5, spsa_epsilon=0.01, damping_coeff=0.5))

# Set up the QNN model
model = tf.keras.models.Sequential([
    ql,
    tf.keras.layers.Dense(num_classes)
])

# Define optimizer and loss
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# Start training with GSPSA
model = train(model=model, train_data=(X_train_tensor, y_train_tensor), val_data=(X_val_tensor, y_val_tensor),
              optimizer=optimizer, loss_fn=loss_fn, num_epochs=num_epochs, method='GSPSA', patience=patience)
plot_prediction(model=model, X_=X_test_tensor, y=y_test_tensor, file_name="examples/qml_tfq_predictions_GSPSA.png")