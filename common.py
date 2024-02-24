import dataclasses
import enum
import typing as t

import numpy as np
import numpy.typing as npt


Predictor = t.Callable[[npt.ArrayLike], npt.ArrayLike]


LearningAlg = t.Callable[[
    # TODO: Depending on how the other regularization techniques work,
    #   we _may_ need to add additional arguments here.
    #   However, this will violate the specification that we were
    #   given, so we should only do this if we are confident that the
    #   specification is wrong.
    npt.ArrayLike, # X
    npt.ArrayLike # Y
], Predictor]


class RegularizationMethod(str, enum.Enum):
    Dropout = 'Dropout'
    NoiseAddition = 'NoiseAddition'
    Robust = 'Robust'


class EvalCriterion(str, enum.Enum):
    MAD = 'MAD'
    MSE = 'MSE'


def mad(yhat: npt.NDArray, y: npt.NDArray) -> float:
    return np.mean(np.abs(y - yhat))


def mse(yhat: npt.NDArray, y: npt.NDArray) -> float:
    residuals = y - yhat
    return np.mean(residuals@residuals)


EVAL_FUNCTIONS = {
    EvalCriterion.MAD: mad,
    EvalCriterion.MSE: mse,
}


@dataclasses.dataclass(frozen=True)
class RegContext:
    learning_alg: LearningAlg
    regularization_method: RegularizationMethod
    eval_criterion: EvalCriterion
    # number of CV folds
    K: int
    # number of MC replicates
    M: t.Optional[int]
    # column bounds for robust
    c: t.Optional[npt.ArrayLike]