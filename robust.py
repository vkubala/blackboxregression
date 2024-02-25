import itertools
import random
import typing as t

import numpy as np
import numpy.typing as npt

from common import EVAL_FUNCTIONS, Predictor, RegContext


def evaluate_c_on_split(
        c: npt.NDArray,
        X_train: npt.NDArray,
        X_test: npt.NDArray,
        Y_train: npt.NDArray,
        Y_test: npt.NDArray,
        reg_context: RegContext,
) -> float:
    predictor = train_model_with_robust(
        c=c,
        X=X_train,
        Y=Y_train,
        reg_context=reg_context,
    )
    Y_hat_test = predictor(X_test)
    return EVAL_FUNCTIONS[reg_context.eval_criterion](
        Y_hat_test,
        Y_test,
    )


def train_model_with_robust(
        c: npt.NDArray,
        X: npt.NDArray,
        Y: npt.NDArray,
        reg_context: RegContext,
) -> Predictor:
    perturbation_matrices = generate_perturbation_matrices(
        X_train=X,
        c=c,
    )
    models: t.List[Predictor] = [
        reg_context.learning_alg(X + delta, Y)
        for delta in perturbation_matrices
    ]
    # also consider the model that is optimal under the unperturbed data
    models.append(reg_context.learning_alg(X, Y))

    # evaluate the loss of each model under each perturbation matrix.
    # compute the maximum loss across perturbation matrices for each model.
    maximum_loss_by_model = [
        max(
            EVAL_FUNCTIONS[reg_context.eval_criterion](
                yhat=model(X + delta),
                y=Y,
            )
            for delta in perturbation_matrices
        )
        for model in models
    ]
    # return model that has the lowest maximum loss
    return models[min(range(len(models)), key=maximum_loss_by_model.__getitem__)]
    

def generate_perturbation_matrices(
    X_train: npt.NDArray,
    c: npt.ArrayLike,
    num_to_generate: int=100,
) -> t.List[npt.NDArray]:
    perturbation_matrices = []
    
    # First half of perturbation matrices add evenly distributed noise
    #  Generate N(0,5) iid RV's
    num_first_set = int(np.ceil(0.5 * num_to_generate))
    for _ in range(num_first_set):
        perturbation_matrix = np.zeros_like(X_train)
        
        for j in range(X_train.shape[1]):
            perturbation = np.random.normal(0, 5, X_train.shape[0])
            norm = np.linalg.norm(perturbation)
            
            #  Normalization to satisfy <c_j ineq. column-wise.
            if norm > c[j]:
                perturbation *= c[j] / norm
            perturbation_matrix[:, j] = perturbation
        perturbation_matrices.append(perturbation_matrix)

    # Second half of perturbation matrices add 5% outliers
    #  Generate N(0,10) iid RV's for few randomly selected rows 
    num_nonzero = int(np.ceil(0.05 * X_train.shape[0]))
    
    if np.linalg.norm(c) != 0:
      num_second_set = num_to_generate - num_first_set
      for _ in range(num_second_set):
          perturbation_matrix = np.zeros_like(X_train)

          for j in range(X_train.shape[1]):
              perturbation = np.zeros(X_train.shape[0])

              #  Randomly select outlier rows.
              nonzero_indices = np.random.choice(X_train.shape[0], num_nonzero, replace=False)
              perturbation[nonzero_indices] = np.random.normal(0, 10, num_nonzero)
              norm = np.linalg.norm(perturbation)

              #  Normalization to satisfy <c_j ineq. column-wise.
              if norm > c[j]:
                  perturbation *= c[j] / norm
              perturbation_matrix[:, j] = perturbation
          perturbation_matrices.append(perturbation_matrix)
    
    return perturbation_matrices


def generate_c_searchspace(
    c: npt.ArrayLike,
    grid_size_max: int = 50,
) -> t.List[npt.NDArray]:
    """Generate the set of column bound vectors over which to perform CV."""
    
    # Sum of non-negative regularization terms in input c
    c_l1 = np.sum(c)  
    
    # Column dimension (number of variables) in training data
    p = len(c)

    # Number of search points each variable gets within bounds c_i
    #  Proportional to the relative size of the bounds
    #  (larger bounds need finer grid)
    rho = np.ceil(4 * p * (c / c_l1)).astype(int)

    # Generate points for each c_i using list comprehension
    search_points = []
    for c_i, num_points in zip(c, rho):

        # If allocated number of search points 1, let that be c_i
        if num_points == 1:
            search_points.append(np.array([c_i]))

        # If allocated number of search points more than, evenly space out
        else:
            points = np.linspace(0, c_i, num_points + 1)[1:-1]
            points = np.append(points, c_i)
            search_points.append(points)

    # Size of the potential searchspace
    combinations_num = np.prod(rho)

    # Limiting the searchspace
    gridsize = min(grid_size_max, combinations_num)

    # If gridsize equals combinations_num, generate all combinations of points
    if gridsize == combinations_num:
        c_vectors = list(itertools.product(*search_points))

    # If number of combinations too large:
    #  randomly pick elements from each array in search_points
    else:
        c_vectors = []
        for _ in range(gridsize):
            vector = [random.choice(points) for points in search_points]
            c_vectors.append(vector)

    return c_vectors