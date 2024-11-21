import mpi4py as mpi
import jax 
import jax.numpy as jnp
import numpy as np

from forecasters.forecasterBase import ForecasterBase


class MPIDistributionStrategy():
    """
    MPIDistributionStrategy is a class that handles the training and forecasting of a model in a distributed environment using MPI.
    Attributes:
        comm (MPI.Comm): The MPI communicator.
        rank (int): The rank of the current process.
        size (int): The total number of processes.
        forecaster (Forecaster): An instance of the Forecaster class.
        local_forecasters (list): A list of local forecasters for each process.
        noise_std (float): The standard deviation of the noise added to the weights and biases during initialization.
        training_completed (bool): A flag indicating whether the training has been completed.
        input_size (list): The size of the input data.
        learnt_weights (list): A list to store the learnt weights after training.
        learnt_biases (list): A list to store the learnt biases after training.
        num_epochs (int): The number of epochs for training.
        W (array-like): The initial weights for the model.
        b (array-like): The initial biases for the model.
        X (array-like): The input data for training.
        y (array-like): The target data for training.
        X_test (array-like): The input data for forecasting.
        predictions (list): A list to store the predictions after forecasting.
        global_predictions (list): A list to store the gathered predictions from all processes.
        global_biases (list): A list to store the gathered biases from all processes.
        global_weights (list): A list to store the gathered weights from all processes.
    Methods:
        __init__(): Initializes the MPIForecaster instance.
        run_training(num_epochs, W, b, X, y): Runs the training process for the forecaster.
        run_forecasting(horizon, X): Runs the forecasting process for the given horizon and input data.
        gather_results(): Collects the predictions, learnt biases, and learnt weights from all processes.
    """

    def __init__(self, ForecasterType: ForecasterBase) -> None:
        self.comm = mpi.MPI.COMM_WORLD
        self.rank : int = self.comm.Get_rank()
        self.size : int = self.comm.Get_size()

        self.forecaster : ForecasterBase = ForecasterType()
        self.local_forecasters : list = None
        self.noise_std : float = 0.1        
        self.training_completed : bool = False
        
        self.input_size : list = None

    def run_training(self, num_epochs:int, W, b, X, y) -> None:
        """
        Runs the training process for the forecaster.
        Parameters:
            num_epochs (int): The number of epochs to run the training for.
            W (array-like): Initial weights for the model.
            b (array-like): Initial biases for the model.
            X (array-like): Input data for training.
            y (array-like): Target data for training.
        Returns:
            None
        Raises:
            AssertionError: If the number of local forecasters or input size is not initialized.
        Notes:
            - This method broadcasts the training data to all processes.
            - Each process initializes weights and biases with added noise and runs the training loop.
            - The learnt weights and biases are stored in the instance variables `learnt_weights` and `learnt_biases`.
            - A barrier is used to synchronize all processes after training.
            - A message is printed when training is completed by the root process.
        """
        
        assert self.local_forecasters is not None, "The number of forecasters was not initialized"
        assert self.input_size is not None, "The input size was not initialized"

        self.learnt_weights = []
        self.learnt_biases = []

        # If the rank is 0, broadcast the training data
        self.num_epochs = self.comm.bcast(num_epochs, root=0)
        self.W = self.comm.bcast(W, root=0)
        self.b = self.comm.bcast(b, root=0)
        self.X = self.comm.bcast(X, root=0)
        self.y = self.comm.bcast(y, root=0)

        sum_up_to_rank = sum(self.local_forecasters[:self.rank])

        for i in range(self.local_forecasters[self.rank]):
            key = jax.random.PRNGKey(sum_up_to_rank + i) 
            W_noise = jax.random.normal(key, self.W.shape) * self.noise_std
            b_noise = jax.random.normal(key, self.b.shape) * self.noise_std 

            W_init = self.W + W_noise
            b_init = self.b + b_noise
            # Then, training is executed
            learnt_W, learnt_b = self.forecaster.training_loop(self.forecaster.get_grad(), self.num_epochs, W_init, b_init, self.X, self.y)
            self.learnt_weights.append(learnt_W)
            self.learnt_biases.append(learnt_b)
        
        self.training_completed = True
        self.comm.Barrier()
        if self.rank == 0:
            print("Training Completed!")

    def run_forecasting(self, horizon, X) -> None:
        """
        Runs the forecasting process for the given horizon and input data.
        This method performs the following steps:
        1. Ensures that the model has been trained and the necessary parameters are initialized.
        2. Broadcasts the forecasting horizon and input data to all processes.
        3. Initializes an empty list to store predictions.
        4. Iterates over the local forecasters assigned to the current process rank.
        5. For each local forecaster, performs the forecasting using the provided horizon, input data, learnt weights, and biases.
        6. Appends each prediction to the predictions list.
        Args:
            horizon (int): The forecasting horizon.
            X (array-like): The input data for forecasting.
        Raises:
            AssertionError: If the model has not been trained, the number of forecasters is not initialized, or the input size is not initialized.
        """

        assert self.training_completed, "You must train the model before inference"
        assert self.local_forecasters is not None, "The number of forecasters was not initialized"
        assert self.input_size > 0, "The input size was not initialized"

        self.horizon = self.comm.bcast(horizon, root=0)
        self.X_test = self.comm.bcast(X, root=0)     
        self.predictions = []   
        
        for i in range(self.local_forecasters[self.rank]):
            prediction = self.forecaster.forecast(self.horizon, self.X_test, self.learnt_weights[i], self.learnt_biases[i])
            self.predictions.append(prediction)
        
    def gather_results(self):
        """
        This method collects the predictions, learnt biases, and learnt weights from all
        processes in a distributed environment using MPI. The
        gathered data is stored in the instance variables `global_predictions`, `global_biases`,
        and `global_weights` respectively.
        Returns:
            tuple: A tuple containing three elements:
            - global_predictions (list): A list of predictions gathered from all processes.
            - global_biases (list): A list of learnt biases gathered from all processes.
            - global_weights (list): A list of learnt weights gathered from all processes.
        """
        
        # Rank 0 gathers all the data
        self.global_predictions = None
        self.global_biases = None
        self.global_weights = None

        self.global_predictions = self.comm.gather(self.predictions, root=0)
        self.global_biases = self.comm.gather(self.learnt_biases, root=0)
        self.global_weights = self.comm.gather(self.learnt_weights, root=0)

        return self.global_predictions, self.global_biases, self.global_weights
