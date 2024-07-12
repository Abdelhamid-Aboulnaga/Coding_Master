import torch
import torch.nn as nn  # Importing the neural network module from PyTorch
import torch.optim as optim  # Importing optimization algorithms from PyTorch
from sklearn.datasets import load_iris  # Importing the Iris dataset from sklearn
from sklearn.model_selection import train_test_split  # Importing function to split data into training and testing sets
from sklearn.preprocessing import StandardScaler  # Importing the scaler for standardizing features
from sklearn.metrics import accuracy_score  # Importing function to calculate accuracy
""" Here, I am trying to classify flowers by their features (petal and sepal dimensions)"""
# Load the Iris dataset
iris = load_iris()
x = iris.data  # Features
y = iris.target  # Target labels

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=37)

# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Convert the numpy arrays to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define the neural network model
class iris_test(nn.Module):
    def __init__(self):
        super(iris_test, self).__init__()
        self.hidden = nn.Linear(4, 10)  # Hidden layer with 4 input features and 10 output features
        self.output = nn.Linear(10, 3)  # Output layer with 10 input features and 3 output features (one for each class)

    def forward(self, x):
        x = torch.relu(self.hidden(x))  # Apply ReLU activation function to the hidden layer's output
        x = self.output(x)  # Get the final output without activation (logits)
        return x

# Initialize the model
model = iris_test()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer with learning rate of 0.01

# Train the model
num_epochs = 1000  # Number of epochs to train the model
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Clear the gradients
    outputs = model(x_train)  # Forward pass: compute the model output
    loss = criterion(outputs, y_train)  # Compute the loss
    loss.backward()  # Backward pass: compute gradients
    optimizer.step()  # Update the model parameters

    # Print the loss every 100 epochs
    if epoch % 100 == 99:
        print(epoch, loss.item())

# Evaluate the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation
    outputs = model(x_test)  # Forward pass: compute the model output for test data
    _, predicted = torch.max(outputs.data, 1)  # Get the predicted class labels
    accuracy = accuracy_score(y_test, predicted)  # Compute the accuracy
    print(f'accuracy = {accuracy*100}%')  # Print the accuracy
    print(predicted)  # Print the predicted labels
