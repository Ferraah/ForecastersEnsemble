import jax
import jax.numpy as jnp
import numpy as np
import mpi4py as mpi

import forecasters.forecasterBase as forecaster

class Forecaster(forecaster.ForecasterBase):
    def __init__(self) -> None:
        self.grad = jax.grad(self.forecast_1step_with_loss)
        pass

    def forecast_1step(self, X, W, b):
        # JAX does not support in-place operations like numpy, so use jax.numpy and functional updates.
        # X = X.copy()  # Copy the input data to avoid modifying the original data
        X_flatten = X.flatten()
        y_next = jnp.dot(W, X_flatten) + b
        return y_next

    def forecast(self, horizon:int, X, W, b):
        result = []

        # Loop over 'horizon' to predict future values
        for t in range(horizon):
            X_flatten = X.flatten()  # Flatten the window for dot product

            # Get the next prediction
            y_next = self.forecast_1step(X_flatten, W, b)

            # Update X by shifting rows and adding the new prediction in the last row
            X = jnp.roll(X, shift=-1, axis=0)  # Shift rows to the left
            X = X.at[-1].set(y_next)  # Update the last row with the new prediction

            # Append the prediction to results
            result.append(y_next)

        return jnp.array(result)


    def forecast_1step_with_loss(self, params:tuple, X, y)->float:
        W, b = params
        y_next = self.forecast_1step(X, W, b)
        return jnp.sum((y_next - y) ** 2)

    ####################################
    # DEFINITION OF THE TRAINING LOOP  #
    #########(###########################

    def training_loop(self, grad:callable, num_epochs:int, W, b, X, y)->tuple:
        for i in range(num_epochs):
            delta = grad((W, b), X, y)
            W -= 0.1 * delta[0]
            b -= 0.1 * delta[1]
        return W, b
    
    def get_grad(self):
        return self.grad

