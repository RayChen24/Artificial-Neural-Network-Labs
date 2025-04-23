import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Perceptron3D:
    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=3)
        self.b_ = np.float_(0.)
        self.errors_ = []

        for _ in range(self.n_iterations):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            if errors == 0:
                break
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


np.random.seed(1)
X = np.random.randn(200, 3)
y = np.where(X[:, 0] + X[:, 1] - X[:, 2] > 0, 1, -1)


ppn = Perceptron3D(learning_rate=0.01, n_iterations=1000)
ppn.fit(X, y)


fig = plt.figure(figsize=(12, 5))


ax = fig.add_subplot(121, projection='3d')
ax.scatter(X[y==1, 0], X[y==1, 1], X[y==1, 2], c='b', marker='o', label='Class 1')
ax.scatter(X[y==-1, 0], X[y==-1, 1], X[y==-1, 2], c='r', marker='s', label='Class -1')


xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 50),
                     np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 50))
z = lambda x,y: (-ppn.w_[0]*x - ppn.w_[1]*y - ppn.b_) / ppn.w_[2]
ax.plot_surface(xx, yy, z(xx,yy), alpha=0.2, color='g')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend(loc='upper left')
ax.set_title('3D Classification')


ax.view_init(elev=20, azim=45)


ax2 = fig.add_subplot(122)
ax2.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Number of misclassifications')
ax2.set_title('Perceptron Convergence Curve')

plt.tight_layout()
plt.show()

print('Weights: %s' % ppn.w_)
print('Bias: %.3f' % ppn.b_)
