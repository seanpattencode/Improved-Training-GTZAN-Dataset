#Keras 

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

# Load the data from the CSV file
data = pd.read_csv('features_30_sec.csv')

# Split the data into features and labels
X = data.iloc[:, 1:-1] # exclude filename and label columns
y = pd.get_dummies(data['label']) # one-hot encode the labels

#Print out data to confirm it
print(X)
print(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Normalize the data. 
#Note that if commented out, aka no normalization,
# the neural network converges to just giving the same response to every 
#example with horrible accuracy no matter how much training time.
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the neural network model
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=58))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
for epoch in range(1000):
    history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    # Get predictions on the test set
    predictions = model.predict(X_test)
    # Get the predicted and true labels for each example
    for i in range(len(predictions)):
        pred_label = predictions[i].argmax()
        true_label = y_test.iloc[i].argmax()
        #print("Example {0}: Prediction = {1}, True Label = {2}".format(i+1, pred_label, true_label))
    #print("Epoch {0}: Loss = {1}, Accuracy = {2}, Validation Loss = {3}, Validation Accuracy = {4}".format(epoch+1, history.history['loss'][0], history.history['accuracy'][0], history.history['val_loss'][0], history.history['val_accuracy'][0]))
    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print('Test accuracy:', accuracy)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', accuracy)