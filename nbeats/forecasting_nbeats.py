import os
from typing import Any
from d3m import container, utils as d3m_utils
from d3m.exceptions import PrimitiveNotFittedError
from d3m.metadata import params
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces.base import Params, CallResult
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
import nbeats

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
    N-BEATS for time series forecasting
    """
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'bd925663-aeaf-4240-9748-cd77dce33819',
            'version': '0.1.0',
            "name": "N-BEATS models for time series forecasting",
            'description': "Pytorch Implementation of N-BEATS. The model is doing local projections to basis "
                           "functions. These functions include \"trends\" (with polynomial functions) and "
                           "\"seasonalities\" (with harmonic functions). The prediction will consist of adding the "
                           "local projections to these basis functions to the last available value in the ts (Naive "
                           "1). The model decomposes the signals successively through different \"blocks\" of a fully "
                           "connected residual NN.",
            'python_path': 'd3m.primitives.time_series_forecasting.nbeats.DeepNeuralNetwork',
            'source': {
                'name': nbeats.__author__,
                'uris': ['https://github.com/autonlab/nbeats'],
                'contact': 'mailto:donghanw@cs.cmu.edu'
            },
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/autonlab/nbeats.git@{git_commit}#egg=nbeats'.format(
                    git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.DEEP_NEURAL_NETWORK,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.TIME_SERIES_FORECASTING,
        },
    )

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
