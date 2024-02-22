import typing as t

import numpy as np
import numpy.typing as npt

from common import EVAL_FUNCTIONS, LearningAlg, Predictor, RegContext


def evaluate_perturbation_matrix_on_split(
        perturbation_matrix: npt.NDArray,
        learning_alg: LearningAlg,
        X_train: npt.NDArray,
        X_test: npt.NDArray,
        Y_train: npt.NDArray,
        Y_test: npt.NDArray,
        reg_context: RegContext,
) -> float:
    predictor = learning_alg(
        X_train + perturbation_matrix,
        Y_train,
    )
    Y_hat_test = predictor(X_test)
    return EVAL_FUNCTIONS[reg_context.eval_criterion](
        Y_hat_test,
        Y_test,
    )


def train_model_with_robust(
        perturbation_matrix: npt.NDArray,
        learning_alg: LearningAlg,
        X: npt.NDArray,
        Y: npt.NDArray,
) -> Predictor:
    assert perturbation_matrix.shape == X.shape
    return learning_alg(
        X + perturbation_matrix,
        Y,
    )


def generate_candidate_perturbations(num: int=100) -> t.List[npt.NDArray]:
    # TODO: Implement this.
    pass