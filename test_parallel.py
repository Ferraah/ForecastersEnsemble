from forecasters.forecaster import *
from mpi4py import *
from mpiDistribution import MPIDistributionStrategy
from util import *

if not MPI.Is_initialized():
    mpi.Init()

X = None
y = None
W = None
b = None

forecaster = MPIDistributionStrategy(Forecaster)

total_forecasters = 1000

local_f = [total_forecasters // forecaster.size for i in range(forecaster.size)]

forecaster.local_forecasters = [local + 1 if i < total_forecasters % forecaster.size else local for (i, local) in enumerate(local_f)] 
if forecaster.rank == 0:
    print(forecaster.local_forecasters)

forecaster.input_size = 2

if forecaster.rank == 0:
    X = jnp.array([[0.1, 0.4], [0.1, 0.5], [0.1, 0.6]]) 
    y = jnp.array([[0.1, 0.7]]) 
    W = jnp.array([[0., 1., 0., 1., 0., 1.], [0., 1., 0, 1., 0., 1.]]) 
    b = jnp.array([0.1]) 

forecaster.run_training(20, W, b, X, y)

forecaster.run_forecasting(5, X)

res, bias, weights = forecaster.gather_results()

if forecaster.rank == 0:
    store_predictions_csv(res, forecaster.input_size, forecaster.horizon, "predictions.csv")
    store_biases_csv(bias, "biases.csv")
    store_weights_csv(weights, forecaster.input_size, 6,"weights.csv")
    print("Execution Completed!")

if not MPI.Is_finalized():
    MPI.Finalize()