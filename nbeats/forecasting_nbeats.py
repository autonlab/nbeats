from typing import Any
from d3m import container, utils as d3m_utils
from d3m.exceptions import PrimitiveNotFittedError
from d3m.metadata import params
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces.base import Params, CallResult
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    pass


class ForecastingNBEATSParams(params.Params):
    is_fitted: bool
    time_column: Any
    integer_time: Any
    filter_idxs: Any
    y_mean: Any
    esrnn: Any


class ForecastingNBEATSHyperparams(hyperparams.Hyperparams):
    pass


class ForecastingNBEATSPrimitive(SupervisedLearnerPrimitiveBase[Inputs, Outputs, ForecastingNBEATSParams,
                                                                ForecastingNBEATSHyperparams]):
    """
    Hybrid ES-RNN models for time series forecasting
    """

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        pass

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        pass

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        pass

    def get_params(self) -> Params:
        pass

    def set_params(self, *, params: Params) -> None:
        pass
