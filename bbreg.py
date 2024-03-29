import typing as t

import numpy as np
import numpy.typing as npt
from sklearn import model_selection

from common import LearningAlg, RegularizationMethod, EvalCriterion, Predictor, RegContext
from noise_addition import Distribution
import dropout
import noise_addition
import robust


def black_box_regress(
        learning_alg: LearningAlg,
        X: npt.ArrayLike,
        Y: npt.ArrayLike,
        regularization_method: RegularizationMethod,
        eval_criterion: EvalCriterion,
        K: int,
        M: t.Optional[int] = None,
        c: t.Optional[npt.ArrayLike] = None,
        verbose: bool = False,
) -> Predictor:
    """Produce a model using the given regularization technique.

    For more information on the types here, see `common.py`.
    
    :param learning_alg: the base learning algorithm
    :param X: the design matrix on which to train
    :param Y: the labels on which to train
    :param regularization_method: the regularization method to use.
        This should be one of the supported options (see RegularizationMethod in common)
        and can be passed in as a string.
    :param eval_criterion: the loss function on which to evaluate the model
        during cross-validation.
        This should be one of the supported options (see EvalCriterion in common)
        and can be passed in as a string.
    :param K: The number of folds to use for cross-validation.
    :param M: The number of Monte Carlo replicates to be used. This is required for
        Dropout and NoiseAddition regularization.
    :param c: A vector of column bounds, required for Robust regularization.
    :param verbose: If this is set to true, then this function will print out the
        parameter setting that was selected during CV.
    
    :returns: A model that takes in new data and outputs predictions. This model
        is trained by `learning_alg`, on data that are modified by the given
        regularization method that was deemed the best during cross-validation.
    """
    X_array = np.asarray(X)
    Y_array = np.asarray(Y)
    _validate_inputs(X_array, Y_array, regularization_method, M, c)
    
    normalizing_vector = np.linalg.norm(X_array, axis=0)
    X_array /= normalizing_vector
    context = RegContext(
        learning_alg=learning_alg,
        regularization_method=regularization_method,
        eval_criterion=eval_criterion,
        K=K,
        M=M,
        c=c,
    )

    if regularization_method == RegularizationMethod.Dropout:
        optimal_dropout_probability = _gridsearch_over_parameters(
            parameter_settings=np.linspace(0, 0.5, 30),
            evaluate_on_split=dropout.evaluate_dropout_probability_on_split,
            X=X_array,
            Y=Y_array,
            reg_context=context,
        )
        if verbose:
            print("Optimal Dropout Probability:", optimal_dropout_probability)
        optimal_model: Predictor = dropout.train_model_with_dropout(
            dropout_prob=optimal_dropout_probability,
            X=X_array,
            Y=Y_array,
            reg_context=context,
        )
    elif regularization_method == RegularizationMethod.NoiseAddition :
        #generate std range based on length of matrix because of euclidean scaling
        std_range = np.linspace(1e-11, 3/np.sqrt(len(Y)), 50)
        # Generate parameter permutations to test, 
        #   explicitly adding (normal, 0) to also test no regularization
        parameter_settings = [(Distribution.normal, 0)] + [
            (distribution, std) for distribution in Distribution
            for std in std_range
        ]
        optimal_distrib, optimal_std = _gridsearch_over_parameters(
            parameter_settings=parameter_settings,       
            evaluate_on_split=noise_addition.evaluate_noise_sd_on_split,
            X=X_array,
            Y=Y_array,
            reg_context=context,
        )
        if verbose:
            print("Optimal Distribution:", optimal_distrib, "Optimal Standard Deviation:", optimal_std)
        optimal_model: Predictor =  noise_addition.train_model_with_noise(
            standard_deviation=optimal_std,
            distribution=optimal_distrib, 
            X=X_array,
            Y=Y_array,
            reg_context=context,
        )
    elif regularization_method == RegularizationMethod.Robust:
        optimal_c = _gridsearch_over_parameters(
            parameter_settings=robust.generate_c_searchspace(c),
            evaluate_on_split=robust.evaluate_c_on_split,
            X=X_array,
            Y=Y_array,
            reg_context=context,
        )
        if verbose:
            print("Optimal c for Robust Regularization:", optimal_c)
        optimal_model: Predictor =  robust.train_model_with_robust(
            c=optimal_c,
            X=X_array,
            Y=Y_array,
            reg_context=context,
        )
    else:
        raise TypeError("{method} is not supported.".format(method=str(regularization_method)))

    # The `optimal_model` computed above expects to take in data that
    #   are normalized. To give the user good performance without needing to
    #   do the normalization theirself, embed the normalization into the model we give them.
    def optimal_model_wrapped_with_normalization(X: npt.NDArray) -> npt.NDArray:
        return optimal_model(X / normalizing_vector)
    return optimal_model_wrapped_with_normalization


def _validate_inputs(
        X: npt.NDArray, Y: npt.NDArray, regularization_method, M, c
    ) -> None:
    """Throw user-friendly errors when the inputs given to
    `black_box_regress` are invalid.
    Inputs are defined in the same way as in `black_box_regress`.
    """
    if len(X.shape) != 2:
        raise ValueError("X must be a 2D array.")
    if len(Y.shape) != 1:
        raise ValueError("Y must be a 1D array.")
    n, _ = X.shape
    if Y.size != n:
        raise ValueError("X and Y must contain the same amount of data.")
    if (
        regularization_method in {RegularizationMethod.Dropout, RegularizationMethod.NoiseAddition}
        and M is None
    ):
        raise ValueError("For the Dropout and NoiseAddition regularization methods,"
            "the number of Monte Carlo replicates `M` is required.")
    if regularization_method == RegularizationMethod.Robust and c is None:
        raise ValueError("For the Robust regularization method,"
            "the vector of column bounds `c` is a required input.")


ParamSetting = t.TypeVar("ParamSetting")
def _gridsearch_over_parameters(
        parameter_settings: t.List[ParamSetting],
        evaluate_on_split: t.Callable[[
            ParamSetting, # parameter setting to evaluate
            npt.NDArray, # X_train
            npt.NDArray, # X_test
            npt.NDArray, # Y_train
            npt.NDArray, # Y_test
            RegContext,
        ], float],
        X: npt.NDArray,
        Y: npt.NDArray,
        reg_context: RegContext,
) -> ParamSetting:
    """Return the optimal setting of the parameters (i.e., the one that achieves the
    least average value of `evaluate_on_split` across folds, among the possibilities
    in `parameter_settings`).
    
    This is general enough to be used for any regularization method and could, without
    too much additional work, be swapped for a different optimization technique."""
    kf = model_selection.KFold(reg_context.K)
    loss_by_fold_by_param_index: t.List[npt.NDArray] = [
        np.zeros(reg_context.K)
        for _ in parameter_settings
    ]
    for fold_number, (train_indices, test_indices) in enumerate(kf.split(X)):
        X_train, X_test = X[train_indices], X[test_indices]
        Y_train, Y_test = Y[train_indices], Y[test_indices]
        for param_index, param_setting in enumerate(parameter_settings):
            loss_by_fold_by_param_index[param_index][fold_number] = evaluate_on_split(
                param_setting,
                X_train,
                X_test,
                Y_train,
                Y_test,
                reg_context,
            )
    mean_loss_by_param_index = [
        np.mean(loss_by_fold)
        for loss_by_fold in loss_by_fold_by_param_index
    ]
    return parameter_settings[
        min(range(len(parameter_settings)), key=mean_loss_by_param_index.__getitem__)
    ]
