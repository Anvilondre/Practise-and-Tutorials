import torch
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


class My_LinearRegression:
    def __init__(
        self,
        solver='equation',
        normal_init=True,
        fit_intercept=True,
        learning_rate=1e-6,
        tolerance=1e-3,
        n_steps=10000,
        verbose=False
    ):
        self.solver = solver
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.normal_init = normal_init
        self.verbose = verbose
        self.tolerance = tolerance
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def cost(self, X, y):
        m = y.size(0)
        y_hat = torch.mm(X, self.W)
        error = (y_hat - y) ** 2
        return 1 / (2 * m) * torch.sum(error)

    def gradient_descent(self, X, y):
        m = X.size(0)
        for i in range(self.n_steps):
            objective = self.cost(X, y)
            grad = torch.mm(X.t(), (torch.mm(X, self.W) - y))
            self.W -= self.learning_rate * 1 / m * grad
            new_error = self.cost(X, y)
            change = torch.abs(torch.sum(objective - new_error))
            if change < self.tolerance:
                break
            if self.verbose and i % 100 == 0:
                print(self.cost(X, y).item())

    def normal_equation(self, X, y):
        det = torch.det(torch.mm(X.t(), X))
        if det == 0:
            raise Exception('X.T * X is a singular matrix')
        self.W = torch.mm(
            torch.inverse(torch.mm(X.t(), X)), torch.mm(X.t(), y)
        )

    def fit(self, X_train, y_train):
        X = torch.from_numpy(X_train).float().to(self.device)
        y = torch.from_numpy(y_train).float().to(self.device)
        y = y.view(-1, 1)

        if self.fit_intercept:
            ones = torch.ones(X.size(0), 1).to(self.device)
            X = torch.cat((ones, X), 1)

        if self.normal_init:
            self.W = torch.rand((X.size(1), 1)).to(self.device)
        else:
            self.W = torch.zeros((X.size(1), 1)).to(self.device)

        if self.solver == 'gr_d':
            self.gradient_descent(X, y)
        elif self.solver == 'normal':
            self.normal_equation(X, y)

    def predict(self, X):
        X = torch.from_numpy(X).float().to(self.device)
        if self.fit_intercept:
            ones = torch.ones(X.size(0), 1).to(self.device)
            X = torch.cat((ones, X), 1)
        return torch.mm(X, self.W).view(-1).cpu().numpy()

    def coef_(self):
        if self.fit_intercept:
            return self.W[1:].view(-1).cpu().numpy()
        else:
            return self.W.view(-1).cpu().numpy()

    def intercept(self):
        if self.fit_intercept:
            return self.W[0].cpu().numpy()
        else:
            return None


if __name__ == '__main__':
    X, y = make_regression(n_samples=10000, n_features=5,
                           n_targets=1, bias=2.5, noise=40, random_state=42)
    print(f'X shape: {X.shape}')
    print(f'y shape: {y.shape}')
    print('=' * 10)

    # my linear regression
    my_model_grad_descent = My_LinearRegression(
        solver='gr_d', learning_rate=1e-3)
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
