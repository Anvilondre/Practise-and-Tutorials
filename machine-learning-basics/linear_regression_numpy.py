import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


class My_LinearRegression:
    def __init__(
            self,
            solver='gr_d',
            normal_init=True,
            fit_intercept=True,
            learning_rate=1e-3,
            n_steps=10000,
            verbose=False):
        self.solver = solver
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.normal_init = normal_init
        self.verbose = verbose
        self.W = None

    def cost(self, X, y):
        m = len(y)
        y_hat = X.dot(self.W)
        error = (y_hat - y) ** 2
        return 1 / (2 * m) * np.sum(error)

    def gradient_descent(self, X, y):
        m = X.shape[0]
        for i in range(self.n_steps):
            grad = np.dot(X.T, (np.dot(X, self.W) - y))
            self.W -= self.learning_rate * 2 / m * grad
            if self.verbose and i % 100 == 0:
                print(self.cost(X, y))

    def normal_equation(self, X, y):
        det = np.linalg.det(np.dot(X.T, X))
        if det == 0:
            raise Exception('X.T * X is a singular matrix')
        self.W = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        if self.normal_init:
            self.W = np.random.rand((X.shape[1]))
        else:
            self.W = np.zeros((X.shape[1]))

        if self.solver == 'gr_d':
            self.gradient_descent(X, y)
        else:
            self.normal_equation(X, y)

    def predict(self, X):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X, self.W)

    def coef_(self):
        if self.fit_intercept:
            return self.W[1:]
        else:
            return self.W

    def intercept(self):
        if self.fit_intercept:
            return self.W[0]
        else:
            None


if __name__ == '__main__':
    X, y = make_regression(n_samples=10000, n_features=5,
                           n_targets=1, bias=2.5, noise=40, random_state=42)
    print(f'X shape: {X.shape}')
    print(f'y shape: {y.shape}')

    # my linear regression
    my_model_grad_descent = My_LinearRegression()
    my_model_grad_descent.fit(X, y)
    grad_descent_preds = my_model_grad_descent.predict(X)
    print(
        f'Grad descent model MSE: {mean_squared_error(y, grad_descent_preds)}')
    print(
        f'Intercept: {my_model_grad_descent.intercept()} \nCoefs: {my_model_grad_descent.coef_()}')
    print('=' * 10)

    # my normal equation
    my_model_normal = My_LinearRegression(solver='normal')
    my_model_normal.fit(X, y)
    normal_preds = my_model_normal.predict(X)
    print(f'Normal Equation MSE: {mean_squared_error(y, normal_preds)}')
    print(
        f'Intercept: {my_model_normal.intercept()} \nCoefs: {my_model_normal.coef_()}')
    print('=' * 10)

    # sklearn model
    sklearn_model = LinearRegression()
    sklearn_model.fit(X, y)
    sklearn_preds = sklearn_model.predict(X)
    print(f'Sklearn MSE: {mean_squared_error(y, sklearn_preds)}')
    print(
        f'Intercept: {sklearn_model.intercept_} \nCoefs: {sklearn_model.coef_}')
