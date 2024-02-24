import enum
import math 
import typing as t

import numpy as np
import numpy.typing as npt

from common import RegContext, EVAL_FUNCTIONS, LearningAlg, Predictor


class Distribution(str, enum.Enum):
    normal = 'normal'
    logistic = 'logistic'
    laplace = 'laplace'


def evaluate_noise_sd_on_split(
        params: t.Tuple[Distribution, float],
        X_train: npt.NDArray,
        X_test: npt.NDArray,
        Y_train: npt.NDArray,
        Y_test: npt.NDArray,
        reg_context: RegContext,
) -> float:
    """Compute the loss (according to the evlauation criterion)
    achieved by the noised (per standard deviation) algorithm on the given folds of data."""
    distribution, standard_deviation = params
    
    assert standard_deviation >= 0
    
    predictor_with_noise = train_model_with_noise(
        standard_deviation=standard_deviation,
        distribution = distribution,
        learning_alg=reg_context.learning_alg,
        X=X_train,
        Y=Y_train,
        mc_replicates=reg_context.M,      
    )
    Y_hat_test = predictor_with_noise(X_test)
    loss_value = EVAL_FUNCTIONS[reg_context.eval_criterion](
        Y_hat_test,
        Y_test,
    )
    return loss_value


def train_model_with_noise(
        standard_deviation: float,
        distribution: Distribution, 
        learning_alg: LearningAlg,
        X: npt.NDArray,
        Y: npt.NDArray,
        mc_replicates: int
) -> Predictor:
    # entry (i, j) of X has noise added from a specified distribution with sd:
    # standard_deviation
    
    X_train_all = []
    Y_train_all = []
    for _ in range(mc_replicates):      
        X_train_noise = X.copy()
        

        noise = generate_noise(distribution, standard_deviation, X_train_noise.shape)   
        
        X_train_noise += noise 
        
        # Create concatenated dataset of M monte carlo noise replicates    
        X_train_all.append(X_train_noise)
        Y_train_all.append(Y)     

    # Concatenate to form a single large training dataset and label set
    X_train_noised = np.concatenate(X_train_all, axis=0)
    Y_train_full = np.concatenate(Y_train_all, axis=0)
    
    return learning_alg(X_train_noised, Y_train_full)


def generate_noise(distribution, standard_deviation, size):
    """
    Generates iid noise with mean 0 from 'normal', 'logistic', or 'laplace' distribution
    specified by standard deviation and size (n, d), where n is samples and d is features.

    Returns: numpy.ndarray of noise values.
    """
    if distribution == Distribution.normal:
        noise = np.random.normal(loc = 0, scale = standard_deviation, size = size)
    elif distribution == Distribution.logistic:
        noise = np.random.logistic(loc = 0, scale = (standard_deviation * np.sqrt(3)) / math.pi , size = size)
    elif distribution == Distribution.laplace:
        noise = np.random.laplace(loc = 0, scale = standard_deviation/np.sqrt(2), size = size)
    else:
        # Not currently needed as user distributional specification is not enabled. Included for future generalizability. 
        raise ValueError(f"Unsupported distribution: {distribution}")
    return noise
