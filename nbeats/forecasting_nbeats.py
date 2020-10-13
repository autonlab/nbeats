import os
import pandas as pd
import torch
from typing import Any
from d3m import container, utils as d3m_utils
from d3m.exceptions import PrimitiveNotFittedError
from d3m.metadata import params
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces.base import Params, CallResult
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
import nbeats
from nbeats.contrib.nbeatsx.nbeatsx import Nbeats

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
    nbeats: Any


class ForecastingNBEATSHyperparams(hyperparams.Hyperparams):
    input_size_multiplier = hyperparams.UniformInt(
        default=2,
        lower=1,
        upper=10000,
        description="",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )
    output_size = hyperparams.UniformInt(
        default=60,
        lower=1,
        upper=10000,
        description="The forecast horizon of the recursive neural network, usually multiple of seasonality. The "
                    "forecast horizon is the number of periods to forecast.",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )
    window_sampling_limit_multiplier = hyperparams.UniformInt(
        default=1,
        lower=1,
        upper=10000,
        description="",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )
    shared_weights = hyperparams.UniformBool(
        default=True,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="",
    )
    stack_types = hyperparams.List(
        elements=hyperparams.Hyperparameter[str](''),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=['trend', 'seasonality'],
        description=""
    )
    n_blocks = hyperparams.List(
        elements=hyperparams.Hyperparameter[int](1),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=[3, 3],
        description=""
    )
    n_layers = hyperparams.List(
        elements=hyperparams.Hyperparameter[int](1),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=[4, 4],
        description=""
    )
    n_hidden = hyperparams.List(
        elements=hyperparams.Hyperparameter[int](1),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=[256, 2048],
        description=""
    )
    n_harmonics = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=1,
        description=""
    )
    n_polynomials = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=2,
        description=""
    )
    learning_rate = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=1e-3,
        description='Size of the stochastic gradient descent steps'
    )
    lr_decay = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=1.0,
        description='The gamma parameter of the RNN scheduler to shrink the learning rate.'
    )
    n_lr_decay_steps = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=3,
        description=""
    )
    batch_size = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=1024,
        description=""
    )
    n_iterations = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=300,
        description=""
    )
    loss = hyperparams.Hyperparameter[str](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default='MAPE',
        description=""
    )
    frequency = hyperparams.Hyperparameter[str](
        default="",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="A number of string aliases are given to useful common time series frequencies. If empty, "
                    "we will try to infer the frequency from the data. If it fails, we use 'D'. "
                    "See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases for a list of frequency aliases",
    )
    seasonality = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=1,
        description=""
    )
    device = hyperparams.Hyperparameter[str](
        default="cpu",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="Specify the device, such as cpu, cuda, cuda:0. We recommend using CPU. It fallbacks to "
                    "CPU if GPU is not available",
    )


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

    def __init__(self, *, hyperparams: ForecastingNBEATSHyperparams, random_seed: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        self._is_fitted = False

        self._device = 'cpu' if not torch.cuda.is_available() or hyperparams['device'] == 'cpu' else hyperparams[
            'device']
        print("Use " + self._device)
        self.logger.info("Use " + self._device)

        self._nbeats = Nbeats(input_size_multiplier=hyperparams['input_size_multiplier'],
                              window_sampling_limit_multiplier=hyperparams['window_sampling_limit_multiplier'],
                              shared_weights=hyperparams['shared_weights'],
                              output_size=hyperparams['output_size'],
                              stack_types=hyperparams['stack_types'],
                              n_blocks=hyperparams['n_blocks'],
                              n_layers=hyperparams['n_layers'],
                              n_hidden=hyperparams['n_hidden'],
                              n_harmonics=hyperparams['n_harmonics'],
                              n_polynomials=hyperparams['n_polynomials'],
                              n_iterations=hyperparams['n_iterations'],
                              learning_rate=hyperparams['learning_rate'],
                              lr_decay=hyperparams['lr_decay'],
                              n_lr_decay_steps=hyperparams['n_lr_decay_steps'],
                              batch_size=hyperparams['batch_size'],
                              loss=hyperparams['loss'],
                              seasonality=hyperparams['seasonality'],
                              random_seed=random_seed,
                              device=self._device)
        self._time_column = None
        self._integer_time = False
        self.filter_idxs = []
        self._year_column = None
        self._constant = 0  # the constant term to avoid nan
        self._y_mean = 0  # the mean of the target variable in the training data

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if not self._is_fitted:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        inputs_copy = inputs.copy()

        # if datetime columns are integers, parse as # of days
        if self._integer_time:
            inputs_copy[self._time_column] = pd.to_datetime(
                inputs_copy[self._time_column], unit="D"
            )
        else:
            inputs_copy[self._time_column] = pd.to_datetime(
                inputs_copy[self._time_column], unit="s"
            )

        # find marked 'GroupingKey' or 'SuggestedGroupingKey'
        grouping_keys = inputs_copy.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/GroupingKey"
        )
        suggested_grouping_keys = inputs_copy.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/SuggestedGroupingKey"
        )
        if len(grouping_keys) == 0:
            grouping_keys = suggested_grouping_keys
        else:
            inputs_copy = inputs_copy.drop(columns=[list(inputs_copy)[i] for i in suggested_grouping_keys])

        # check whether no grouping keys are labeled
        if len(grouping_keys) == 0:
            concat = pd.concat([inputs_copy[self._time_column]], axis=1)
            concat.columns = ['ds']
            concat['unique_id'] = 'series1'  # We have only one series
        else:
            # concatenate columns in `grouping_keys` to unique_id column
            concat = inputs_copy.loc[:, self.filter_idxs].apply(lambda x: ' '.join([str(v) for v in x]), axis=1)
            concat = pd.concat([concat, inputs_copy[self._time_column]], axis=1)
            concat.columns = ['unique_id', 'ds']

        X_test = concat[['unique_id', 'ds']]

        predictions = self._nbeats.predict(X_test)
        predictions['y_hat'] -= self._constant
        predictions['y_hat'] = self._fillna(predictions['y_hat'])
        output = container.DataFrame(predictions['y_hat'], generate_metadata=True)
        return base.CallResult(output)

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        data = inputs.horizontal_concat(outputs)
        data = data.copy()

        # mark datetime column
        times = data.metadata.list_columns_with_semantic_types(
            (
                "https://metadata.datadrivendiscovery.org/types/Time",
                "http://schema.org/DateTime",
            )
        )
        if len(times) != 1:
            raise ValueError(
                f"There are {len(times)} indices marked as datetime values. Please only specify one"
            )
        self._time_column = list(data)[times[0]]

        # if datetime columns are integers, parse as # of days
        if (
                "http://schema.org/Integer"
                in inputs.metadata.query_column(times[0])["semantic_types"]
        ):
            self._integer_time = True
            data[self._time_column] = pd.to_datetime(
                data[self._time_column], unit="D"
            )
        else:
            data[self._time_column] = pd.to_datetime(
                data[self._time_column], unit="s"
            )

        # sort by time column
        data = data.sort_values(by=[self._time_column])

        # mark key and grp variables
        self.key = data.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/PrimaryKey"
        )

        # mark target variables
        self._targets = data.metadata.list_columns_with_semantic_types(
            (
                "https://metadata.datadrivendiscovery.org/types/SuggestedTarget",
                "https://metadata.datadrivendiscovery.org/types/TrueTarget",
                "https://metadata.datadrivendiscovery.org/types/Target",
            )
        )
        self._target_types = [
            "i"
            if "http://schema.org/Integer"
               in data.metadata.query_column(t)["semantic_types"]
            else "c"
            if "https://metadata.datadrivendiscovery.org/types/CategoricalData"
               in data.metadata.query_column(t)["semantic_types"]
            else "f"
            for t in self._targets
        ]
        self._targets = [list(data)[t] for t in self._targets]

        self.target_column = self._targets[0]

        # see if 'GroupingKey' has been marked
        # otherwise fall through to use 'SuggestedGroupingKey'
        grouping_keys = data.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/GroupingKey"
        )
        suggested_grouping_keys = data.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/SuggestedGroupingKey"
        )
        if len(grouping_keys) == 0:
            grouping_keys = suggested_grouping_keys
            drop_list = []
        else:
            drop_list = suggested_grouping_keys

        grouping_keys_counts = [
            data.iloc[:, key_idx].nunique() for key_idx in grouping_keys
        ]
        grouping_keys = [
            group_key
            for count, group_key in sorted(zip(grouping_keys_counts, grouping_keys))
        ]
        self.filter_idxs = [list(data)[key] for key in grouping_keys]

        # drop index
        data.drop(
            columns=[list(data)[i] for i in drop_list + self.key], inplace=True
        )

        # check whether no grouping keys are labeled
        if len(grouping_keys) == 0:
            concat = pd.concat([data[self._time_column], data[self.target_column]], axis=1)
            concat.columns = ['ds', 'y']
            concat['unique_id'] = 'series1'  # We have only one series
        else:
            # concatenate columns in `grouping_keys` to unique_id column
            concat = data.loc[:, self.filter_idxs].apply(lambda x: ' '.join([str(v) for v in x]), axis=1)
            concat = pd.concat([concat,
                                data[self._time_column],
                                data[self.target_column]],
                               axis=1)
            concat.columns = ['unique_id', 'ds', 'y']

        if len(grouping_keys):
            # Infer frequency
            freq = self._nbeats.mc.frequency
            if not freq:
                freq = pd.infer_freq(concat.head()['ds'])
                if freq is None and len(concat['unique_id']) > 0:
                    freq = pd.infer_freq(concat[concat['unique_id'] == concat['unique_id'][0]]['ds'])
                if freq is None:
                    freq = 'D'
                    self.logger.warn('Cannot infer frequency. Use "D".')
                else:
                    self.logger.info('Inferred frequency: {}'.format(freq))

            # Series must be complete in the frequency
            concat = ForecastingNBEATSPrimitive._ffill_missing_dates_per_serie(concat, freq)

        # remove duplicates
        concat = concat.drop_duplicates(['unique_id', 'ds'])

        self._data = concat

        self._y_mean = self._data['y'].mean()

        # if min of y is negative, then add the absolute value of it to the constant
        if self._data['y'].min() <= 0:
            self._constant = 1 - self._data['y'].min()

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        X_train = self._data[['unique_id', 'ds']]
        X_train['x'] = '1'
        y_train = self._data[['unique_id', 'ds', 'y']]
        y_train['y'] += self._constant
        self._nbeats.fit(X_train, y_train, verbose=False)
        self._is_fitted = True

        return base.CallResult(None)

    def get_params(self) -> Params:
        return ForecastingNBEATSParams(is_fitted=self._is_fitted,
                                       time_column=self._time_column,
                                       integer_time=self._integer_time,
                                       filter_idxs=self.filter_idxs,
                                       y_mean=self._y_mean,
                                       nbeats=self._nbeats)

    def set_params(self, *, params: Params) -> None:
        self._is_fitted = params['is_fitted']
        self._time_column = params['time_column']
        self._integer_time = params['integer_time']
        self.filter_idxs = params['filter_idxs']
        self._y_mean = params['y_mean']
        self._nbeats = params['nbeats']

    @staticmethod
    def _ffill_missing_dates_particular_serie(serie, min_date, max_date, freq):
        date_range = pd.date_range(start=min_date, end=max_date, freq=freq)
        unique_id = serie['unique_id'].unique()
        df_balanced = pd.DataFrame({'ds': date_range, 'key': [1] * len(date_range), 'unique_id': unique_id[0]})

        # Check balance
        check_balance = df_balanced.groupby(['unique_id']).size().reset_index(name='count')
        assert len(set(check_balance['count'].values)) <= 1
        df_balanced = df_balanced.merge(serie, how="left", on=['unique_id', 'ds'])

        df_balanced['y'] = df_balanced['y'].fillna(method='ffill')

        return df_balanced

    @staticmethod
    def _ffill_missing_dates_per_serie(df, freq="D", fixed_max_date=None):
        """Receives a DataFrame with a date column and forward fills the missing gaps in dates, not filling dates before


        Parameters
        ----------
        df: DataFrame
            Input DataFrame
        key: str or list
            Name(s) of the column(s) which make a unique time series
        date_col: str
            Name of the column that contains the time column
        freq: str
            Pandas time frequency standard strings, like "W-THU" or "D" or "M"
        numeric_to_fill: str or list
            Name(s) of the columns with numeric values to fill "fill_value" with
        """
        if fixed_max_date is None:
            df_max_min_dates = df[['unique_id', 'ds']].groupby('unique_id').agg(['min', 'max']).reset_index()
        else:
            df_max_min_dates = df[['unique_id', 'ds']].groupby('unique_id').agg(['min']).reset_index()
            df_max_min_dates['max'] = fixed_max_date

        df_max_min_dates.columns = df_max_min_dates.columns.droplevel()
        df_max_min_dates.columns = ['unique_id', 'min_date', 'max_date']

        df_list = []
        for index, row in df_max_min_dates.iterrows():
            df_id = df[df['unique_id'] == row['unique_id']]
            df_id = ForecastingNBEATSPrimitive._ffill_missing_dates_particular_serie(df_id, row['min_date'],
                                                                                     row['max_date'], freq)
            df_list.append(df_id)

        df_dates = pd.concat(df_list).reset_index(drop=True).drop('key', axis=1)[['unique_id', 'ds', 'y']]

        return df_dates

    def _fillna(self, series):
        if series.isnull().any():
            # self.logger.warning("The prediction contains NAN. Fill with mean of prediction.")
            tofill = series.mean()  # use the prediction mean if possible. Otherwise use the mean of the training data.
            if pd.isna(tofill):
                # self.logger.warn('The predictions are all NAN')
                tofill = self._y_mean
            return series.fillna(tofill)
        return series
