from abc import ABC, abstractmethod
class ForecasterBase(ABC):
    """
    ForecasterBase is a base class for forecasting models.
    It does not implement methods but defines a template for the types that inherit from it

    Methods:
        None
    """
    @abstractmethod
    def forecast_1step(self, X, W, b):
        pass

    @abstractmethod
    def forecast(self, horizon, X, W, b):
        pass

    @abstractmethod
    def forecast_1step_with_loss(self, params:tuple, X, y)->float:
        pass
    
    @abstractmethod
    def training_loop(self, grad:callable, num_epochs:int, W, b, X, y)->tuple:
        pass

    @abstractmethod
    def get_grad(self)->callable:
        pass