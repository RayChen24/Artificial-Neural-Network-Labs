import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Iris.csv')
sc=StandardScaler()
data= data[['PetalLengthCm', 'PetalWidthCm', 'Species']].dropna()
#print(data.to_string())
#print(data.head())

x_train,x_test,y_train,y_test=train_test_split(data[['PetalLengthCm','PetalWidthCm']], data.Species,test_size=0.2,random_state=0)
X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)

class Dataloader:
    def __init__(self, X, y):
        self.X = X
        self.y = pd.get_dummies(y).values  # One-hot encode the labels

    def get_data(self):
        return self.X, self.y

class MLP: 
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)
        
        self.a1_error = self.output_delta.dot(self.W2.T)
        self.a1_delta = self.a1_error * self.sigmoid_derivative(self.a1)
        
        self.W2 += self.a1.T.dot(self.output_delta) * self.learning_rate
        self.b2 += np.sum(self.output_delta, axis=0, keepdims=True) * self.learning_rate
        self.W1 += X.T.dot(self.a1_delta) * self.learning_rate
        self.b1 += np.sum(self.a1_delta, axis=0, keepdims=True) * self.learning_rate
    
    def train(self, X, y, epochs=10000, learning_rate=0.01):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                predictions = (output > 0.5).astypions = (final_output > 0.5).astype(int)
        final_accuracy = np.mean(final_predictions == y)
        print(f'Final Accuracy: {final_accuracy}')

# Create dataloaders for training and testing data
train_loader = Dataloader(X_train, y_train)
test_loader = Dataloader(X_test, y_test)

X_train, y_train = train_loader.get_data()
X_test, y_test = test_loader.get_data()

# Initialize the MLP model
input_size = X_train.shape[1]
hidden_size = 5  # You can adjust this
output_size = y_train.shape[1]
learning_rate = 0.01

mlp = MLP(input_size, hidden_size, output_size, learning_rate)

# Train the model
mlp.train(X_train, y_train, epochs=10000, learning_rate=0.01)

# Print final accuracy
#final_output = mlp.forward(X_test)
#final_predictions = (final_output > 0.5).astype(int)
#final_accuracy = np.mean(final_predictions == y_test)
#print(f'Test Accuracy: {final_accuracy}')