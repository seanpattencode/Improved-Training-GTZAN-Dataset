#Implementation from scratch, no randomization

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the data from the CSV file
data = pd.read_csv('features_30_sec.csv')

# Split the data into features and labels
X = data.iloc[:, 1:-1].to_numpy() # exclude filename and label columns
y = pd.get_dummies(data['label']).to_numpy() # one-hot encode the labels

# Print out data to confirm it
print(X)
print(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the neural network model
def neural_network(input_dim):
    model = {}
    # Initialize parameters for the first layer
    model['W1'] = np.random.randn(input_dim, 128) * 0.01
    model['b1'] = np.zeros((1, 128))
    # Initialize parameters for the second layer
    model['W2'] = np.random.randn(128, 64) * 0.01
    model['b2'] = np.zeros((1, 64))
    # Initialize parameters for the third layer
    model['W3'] = np.random.randn(64, 32) * 0.01
    model['b3'] = np.zeros((1, 32))
    # Initialize parameters for the output layer
    model['W4'] = np.random.randn(32, 10) * 0.01
    model['b4'] = np.zeros((1, 10))
    return model

# Define the forward propagation function
def forward_propagation(X, model):
    # First layer
    Z1 = np.dot(X, model['W1']) + model['b1']
    A1 = np.maximum(Z1, 0)
    # Second layer
    Z2 = np.dot(A1, model['W2']) + model['b2']
    A2 = np.maximum(Z2, 0)
    # Third layer
    Z3 = np.dot(A2, model['W3']) + model['b3']
    A3 = np.maximum(Z3, 0)
    # Output layer
    Z4 = np.dot(A3, model['W4']) + model['b4']
    exp_scores = np.exp(Z4)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Save intermediate values in dictionary
    model['A1'] = A1
    model['A2'] = A2
    model['A3'] = A3
    return probs


# Define the cross-entropy loss function
def cross_entropy_loss(probs, y):
    m = y.shape[0]
    loss = -1/m * np.sum(y * np.log(probs))
    return loss

# Define the backward propagation function
def backward_propagation(X, y, model, probs, learning_rate):
    m = y.shape[0]
    # Output layer
    delta4 = probs - y
    dW4 = 1/m * np.dot(model['A3'].T, delta4)
    db4 = 1/m * np.sum(delta4, axis=0, keepdims=True)
    # Third layer
    delta3 = np.dot(delta4, model['W4'].T) * (model['A3'] > 0)
    dW3 = 1/m * np.dot(model['A2'].T, delta3)
    db3 = 1/m * np.sum(delta3, axis=0, keepdims=True)
    # Second layer
    delta2 = np.dot(delta3, model['W3'].T) * (model['A2'] > 0)
    dW2 = 1/m * np.dot(model['A1'].T, delta2)
    db2 = 1/m * np.sum(delta2, axis=0, keepdims=True)
    # First layer
    delta1 = np.dot(delta2, model['W2'].T) * (model['A1'] > 0)
    dW1 = 1/m * np.dot(X.T, delta1)
    db1 = 1/m * np.sum(delta1, axis=0, keepdims=True)
    # Update parameters
    model['W1'] -= learning_rate * dW1
    model['b1'] -= learning_rate * db1
    model['W2'] -= learning_rate * dW2
    model['b2'] -= learning_rate * db2
    model['W3'] -= learning_rate * dW3
    model['b3'] -= learning_rate * db3
    model['W4'] -= learning_rate * dW4
    model['b4'] -= learning_rate * db4

#Train the neural network model
input_dim = X_train.shape[1]
num_classes = y_train.shape[1]
model = neural_network(input_dim)

learning_rate = 0.1
num_epochs = 100000

for epoch in range(num_epochs):
  # Forward propagation
  probs = forward_propagation(X_train, model)
  # Backward propagation
  backward_propagation(X_train, y_train, model, probs, learning_rate)
  # Print the loss every 100 epochs
  if epoch % 100 == 0:
    loss = cross_entropy_loss(probs, y_train)
    print(f"Epoch {epoch}, Loss: {loss}")

    #Evaluate the model on the test set
    probs = forward_propagation(X_test, model)
    accuracy = np.mean(np.argmax(probs, axis=1) == np.argmax(y_test, axis=1))
    print(f"Test Accuracy: {accuracy}")


#Evaluate the model on the test set
probs = forward_propagation(X_test, model)
accuracy = np.mean(np.argmax(probs, axis=1) == np.argmax(y_test, axis=1))
print(f"Test Accuracy: {accuracy}")





   
