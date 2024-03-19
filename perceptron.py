import numpy as np
import matplotlib.pyplot as plt


def plot(hold=False):
    if W[1] == 0:
        plt.clf()
        y = np.arange(-2, 2, 0.1)
        x = np.full_like(y, -W[2] / W[0])
    else:
        m = -W[0] / W[1]
        b = -W[2] / W[1]
        x = np.arange(-2, 2, 0.1)
        y = m * x + b
    plt.clf()
    plt.plot(x, y, 'b')
    for i in range(len(X)):
        color = 'og' if Y[i] == 1 else 'or'
        plt.plot(X[i][0], X[i][1], color)
    if hold:
        plt.show()
    else:
        plt.pause(1)

# AND
X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
Y = np.array([1, -1, -1, -1])

X_ = np.hstack((X, np.ones((len(X), 1))))
X_ = np.array([X_[i] * Y[i] for i in range(len(X))])
print(X_)

W = X_[np.random.randint(len(X))].copy()
plot()
while True:
    flag = True
    for i, x in enumerate(X_):
        if np.sum(x * W) <= 0:
            W = W + x
            plot()
            flag = False
    if flag:
        break
plot(True)