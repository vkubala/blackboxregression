import numpy as np
import numpy.typing as npt

from common import RegContext, EVAL_FUNCTIONS, LearningAlg, Predictor


def evaluate_noise_sd_on_split(
        standard_deviation: float,
        X_train: npt.NDArray,
        X_test: npt.NDArray,
        Y_train: npt.NDArray,
        Y_test: npt.NDArray,
        reg_context: RegContext,
) -> float:
    """Compute the loss (according to the evlauation criterion)
    achieved by the noised (per standard deviation) algorithm on the given folds of data."""
    assert standard_deviation >= 0

    mc_replicates = reg_context.M
    
    predictor_with_noise = train_model_with_noise(
        standard_deviation=standard_deviation,
        learning_alg=reg_context.learning_alg,
        X=X_train,
        Y=Y_train,
        mc_replicates = reg_context.M,
            
    )
    Y_hat_test = predictor_with_noise(X_test)
    loss_value = EVAL_FUNCTIONS[reg_context.eval_criterion](
        Y_hat_test,
        Y_test,
        )
    return loss_value

#starting with only gaussian noise. We can add additional distributions to generate noise if we want to.
def train_model_with_noise(
        standard_deviation: float,
        learning_alg: LearningAlg,
        X: npt.NDArray,
        Y: npt.NDArray,
        mc_replicates: int
) -> Predictor:
    # This is a matrix whose component (i, j) is the proposition that
    #   entry (i, j) of X has noise added from a gaussian distribution with sd:
    # standard_deviation
    mc_replicates = reg_context.M
    
    X_train_all = []
    Y_train_all = []    
    
    for r in range(mc_replicates):      
        #make copy of X_train so we don't overwrite original X_train with dropout mask
        X_train_noise = X_train.copy()
        
        # Generate an array of independent Gaussian noise values with mean 0 and the specified standard deviation, 
        # matching the dimensions (n x d) of X_train, where n is the number of samples and d is the number of features.
        noise = np.random.normal(loc=0, scale=standard_deviation, size=X_train_noise.shape)
    
        # Add the noise to X_train_noise
        X_train_noise += noise 
        
        #Create concatenated dataset of M monte carlo noise replicates    
        X_train_all.append(X_train_noise)
        Y_train_all.append(Y_train)     

    # Concatenate to form a single large training dataset and label set
    X_train_noised = np.concatenate(X_train_all, axis=0)
    Y_train_full = np.concatenate(Y_train_all, axis=0)
    
    return learning_alg(X_train_noised, Y_train_full)
