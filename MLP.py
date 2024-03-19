import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
Y = np.array([0, 1, 1, 0])[:, None]  # XOR

d = [2, 10, 1]
lr = 0.01
# initial weights
W = []
for i in range(len(d) - 1):
    W.append(np.random.rand(d[i] + 1, d[i + 1]))
inputs = []

def pad(x):
    return np.hstack((x, np.ones((len(x), 1))))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return x * (1 - x)

def forward_(x):
    for i in range(len(W)):
        x = sigmoid(pad(x) @ W[i])
    return x

def forward(x):
    inputs.append(pad(x).copy())
    for i in range(len(W)):
        o = sigmoid(inputs[-1] @ W[i])
        inputs.append(pad(o).copy())
    return o

def backward():
    epoch = 0
    dW = [0] * len(W)
    while True:
        epoch += 1
        output = forward(X)
        error = Y - output
        mse = np.mean(error ** 2)
        if epoch % 5000 == 0:
            print(f'epoch: {epoch} loss: {mse} output:{output.flatten().round()}')

        # output layer
        delta = d_sigmoid(output) * error
        dW[-1] = lr * inputs[-2].T @ delta
        W[-1] += dW[-1]
        # hidden layers
        for i in range(len(W) - 2, -1, -1):
            delta = d_sigmoid(inputs[i + 1]) * (delta @ W[i + 1].T)
            delta = delta[:, :-1]
            dW[i] = lr * inputs[i].T @ delta
            W[i] += dW[i]

backward()