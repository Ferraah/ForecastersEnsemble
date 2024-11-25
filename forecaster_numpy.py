import numpy as np

class NumpyForecaster :
    def forecast_1step(self, X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        X_flatten = X.flatten()
        y_next = np.dot(W, X_flatten) + b
        return y_next

    def forecast(self, horizon: int, X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        result = []
        for t in range(horizon):
            X_flatten = X.flatten()
            y_next = self.forecast_1step(X_flatten, W, b)
            
            # Update X by shifting rows and adding the new prediction in the last row
            X = np.roll(X, shift=-1, axis=0)  # Shift rows to the left
            X[-1] = y_next  # Update the last row with the new prediction

            result.append(y_next)
        
        return np.array(result)

    def forecast_1step_with_loss(self, params: tuple, X: np.ndarray, y: np.ndarray) -> float:
        W, b = params
        y_next = self.forecast_1step(X, W, b)
        # Loss is the squared difference between the expected output and the predicted output
        return np.sum((y_next - y) ** 2)

    # Gradient computation manually using finite differences (since we are not using JAX's grad)
    def compute_gradients(self, W: np.ndarray, b: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple:
        epsilon = 1e-5  # Small perturbation for numerical gradients
        W_grad = np.zeros_like(W)
        b_grad = np.zeros_like(b)
        
        # Gradient with respect to W
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_perturb = W.copy()
                W_perturb[i, j] += epsilon
                loss_plus = self.forecast_1step_with_loss((W_perturb, b), X, y)
                loss_minus = self.forecast_1step_with_loss((W, b), X, y)
                W_grad[i, j] = (loss_plus - loss_minus) / epsilon
        
        # Gradient with respect to b
        for i in range(b.shape[0]):
            b_perturb = b.copy()
            b_perturb[i] += epsilon
            loss_plus = self.forecast_1step_with_loss((W, b_perturb), X, y)
            loss_minus = self.forecast_1step_with_loss((W, b), X, y)
            b_grad[i] = (loss_plus - loss_minus) / epsilon

        return W_grad, b_grad

    # The training loop using gradient descent
    def training_loop(self, grad_func: callable, num_epochs: int, W: np.ndarray, b: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple:
        learning_rate = 0.1
        for epoch in range(num_epochs):
            W_grad, b_grad = grad_func(W, b, X, y)  # Compute gradients
            W -= learning_rate * W_grad  # Update weights
            b -= learning_rate * b_grad  # Update bias
        return W, b 