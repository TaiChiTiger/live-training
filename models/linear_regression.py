import numpy as np

class LinearRegressionFromScrach:
    def __init__(self, learning_rate, epochs, random_state):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

    def mean_squared_loss(self, X, W, y):
        y_pred = X @ W
        return np.sum((y - y_pred)**2) / len(y)

    def negative_gradient(self, X, W, y):
        grad = -2 / len(y) * X.T @ (y - X @ W) 
        return -grad

    def gradient_descent(self, X, y):
        np.random.seed(self.random_state)
        p = X.shape[1]
        W = np.random.randn(p) # 随机初始化权重
        weight_history = [W.copy()]
        loss = self.mean_squared_loss(X, W, y)
        loss_history = [loss]
        tol = 1e-8
        for e in range(1, self.epochs+1):
            # if e == 1 or e % 10 == 0:
            #     print("Epoch {}: mse={}".format(e, loss))
            neg_grad = self.negative_gradient(X, W, y)
            W += self.learning_rate * neg_grad # 更新权重
            weight_history.append(W.copy())
            weight_change = np.sum(np.abs(np.abs(weight_history[-1]) - \
                                        np.abs(weight_history[-2])))
            if weight_change < tol:
                break
            loss = self.mean_squared_loss(X, W, y)
            loss_history.append(loss)
            
        return weight_history, loss_history

