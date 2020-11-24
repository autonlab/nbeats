import os
import time
import numpy as np
import pandas as pd
import random
from collections import defaultdict

import torch as t
from torch import optim

from src.nbeatsx.nbeatsx_model import NBeats, NBeatsBlock, IdentityBasis, TrendBasis, SeasonalityBasis
from src.utils.pytorch.sampler import TimeseriesDataset
from src.utils.pytorch.losses import MAPELoss, MASELoss, SMAPELoss, MSELoss, MAELoss

class Nbeats(object):
    """
    Future documentation
    """
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    IDENTITY_BLOCK = 'identity'

    def __init__(self,
                 input_size_multiplier=2,
                 output_size=1,
                 window_sampling_limit_multiplier=1,
                 shared_weights=True,
                 stack_types=[TREND_BLOCK, SEASONALITY_BLOCK],
                 n_blocks=[3,3],
                 n_layers=[4,4],
                 n_hidden=[256, 2048],
                 n_harmonics=1,
                 n_polynomials=2,
                 batch_normalization=False,
                 dropout_prob=0,
                 weight_decay=0,
                 learning_rate=0.001,
                 lr_decay=0.5,
                 n_lr_decay_steps=3,
                 batch_size=1024,
                 n_iterations=300,
                 loss='MAPE',
                 frequency=None,
                 seasonality=1,
                 random_seed=1,
                 device=None):
        super(Nbeats, self).__init__()

        self.input_size = int(input_size_multiplier*output_size)
        self.output_size = output_size
        self.window_sampling_limit = int(window_sampling_limit_multiplier*output_size)
        self.shared_weights = shared_weights
        self.stack_types = stack_types
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_harmonics = n_harmonics
        self.n_polynomials = n_polynomials
        self.batch_normalization = batch_normalization
        self.dropout_prob = dropout_prob
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.n_lr_decay_steps = n_lr_decay_steps
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.loss = loss
        self.frequency = frequency
        self.seasonality = seasonality
        self.random_seed = random_seed
        if device is None:
            device = 'cuda' if t.cuda.is_available() else 'cpu'
        self.device = device

        self._is_data_parsed = False
        self._eval_criterions = ['MAPE']

    def create_stack(self):
        #print(f'| N-Beats')
        block_list = []
        for i in range(len(self.stack_types)):
            #print(f'| --  Stack {self.stack_types[i]} (#{i})')
            for block_id in range(self.n_blocks[i]):

                 # Batch norm only on first block
                if (len(block_list)==0) and (self.batch_normalization):
                    batch_normalization_block = True
                else:
                    batch_normalization_block = False
                
                # Shared weights
                if self.shared_weights and block_id>0:
                    nbeats_block = block_list[-1]

                else:
                    if self.stack_types[i] == 'seasonality': #NBeats.SEASONALITY_BLOCK
                        nbeats_block = NBeatsBlock(n_inputs=self.input_size,
                                                #    n_static=self.n_static,
                                                theta_dim=4 * int(
                                                        np.ceil(self.n_harmonics / 2 * self.output_size) - (self.n_harmonics - 1)),
                                                basis=SeasonalityBasis(harmonics=self.n_harmonics,
                                                                                backcast_size=self.input_size,
                                                                                forecast_size=self.output_size),
                                                n_layers=self.n_layers[i],
                                                n_hidden=self.n_hidden[i],
                                                batch_normalization=batch_normalization_block,
                                                dropout_prob=self.dropout_prob)
                    elif self.stack_types[i] == 'trend': # NBeats.TREND_BLOCK
                        nbeats_block = NBeatsBlock(n_inputs=self.input_size,
                                                #    n_static=self.n_static,
                                                theta_dim=2 * (self.n_polynomials + 1),
                                                basis=TrendBasis(degree_of_polynomial=self.n_polynomials,
                                                                            backcast_size=self.input_size,
                                                                            forecast_size=self.output_size),
                                                n_layers=self.n_layers[i],
                                                n_hidden=self.n_hidden[i],
                                                batch_normalization=batch_normalization_block,
                                                dropout_prob=self.dropout_prob)
                    elif self.stack_types[i] == 'identity': #NBeats.GENERIC_BLOCK
                        nbeats_block = NBeatsBlock(n_inputs=self.input_size,
                                                #    n_static=self.n_static,
                                                theta_dim=self.input_size + self.output_size,
                                                basis=IdentityBasis(backcast_size=self.input_size,
                                                                    forecast_size=self.output_size),
                                                n_layers=self.n_layers[i],
                                                n_hidden=self.n_hidden[i],
                                                batch_normalization=batch_normalization_block,
                                                dropout_prob=self.dropout_prob)
                #print(f'     | -- {nbeats_block}')
                block_list.append(nbeats_block)
        return block_list

    def __loss_fn(self, loss_name: str):
        def loss(x, freq, forecast, target, mask):
            if loss_name == 'MAPE':
                return MAPELoss(y=target, y_hat=forecast, mask=mask)
            elif loss_name == 'MASE':
                return MASELoss(y=target, y_hat=forecast, y_insample=x, seasonality=freq, mask=mask)
            elif loss_name == 'SMAPE':
                return SMAPELoss(y=target, y_hat=forecast, mask=mask)
            elif loss_name == 'MSE':
                return MSELoss(y=target, y_hat=forecast, mask=mask)
            elif loss_name == 'MAE':
                return MAELoss(y=target, y_hat=forecast, mask=mask)
            else:
                raise Exception(f'Unknown loss function: {loss_name}')
        return loss

    def to_tensor(self, x: np.ndarray) -> t.Tensor:
        tensor = t.as_tensor(x, dtype=t.float32).to(self.device)
        return tensor

    def transform_data(self, y_df, X_s_df, X_t_df):
        unique_ids = y_df['unique_id'].unique()

        if X_t_df is not None:
            X_t_vars = [col for col in X_t_df.columns if col not in ['unique_id','ds']]
        else:
            X_t_vars = []

        if X_s_df is not None:
            X_s_vars = [col for col in X_s_df.columns if col not in ['unique_id']]
        else:
            X_s_vars = []

        ts_data = []
        static_data = []
        meta_data = []
        for i, u_id in enumerate(unique_ids):
            top_row = np.asscalar(y_df['unique_id'].searchsorted(u_id, 'left'))
            bottom_row = np.asscalar(y_df['unique_id'].searchsorted(u_id, 'right'))
            serie = y_df[top_row:bottom_row]['y'].values
            last_ds_i = y_df[top_row:bottom_row]['ds'].max()
            
            # Y values
            ts_data_i = {'y': serie}
            # X_t values
            for X_t_var in X_t_vars:
                serie =  X_t_df[top_row:bottom_row][X_t_var].values
                ts_data_i[X_t_var] = serie
            ts_data.append(ts_data_i)

            # Static data
            s_data_i = defaultdict(list)
            for X_s_var in X_s_vars:
                s_data_i[X_s_var] = X_s_df.loc[X_s_df['unique_id']==u_id, X_s_var].values
            static_data.append(s_data_i)

            # Metadata
            meta_data_i = {'unique_id': u_id,
                           'last_ds': last_ds_i}
            meta_data.append(meta_data_i)

        return ts_data, static_data, meta_data

    def parse_data(self, y_df, X_s_df, X_t_df, offset):
        assert type(y_df) == pd.core.frame.DataFrame
        assert all([(col in y_df) for col in ['unique_id', 'ds', 'y']])
        if X_t_df is not None:
            assert type(X_t_df) == pd.core.frame.DataFrame
            assert all([(col in X_t_df) for col in ['unique_id', 'ds']])

        if self.frequency is None:
            self.frequency = pd.infer_freq(y_df.head()['ds'])
            print("Infered frequency: {}".format(self.frequency))

        #X_t_df, y_df = self.fill_series(X_t_df, y_df) #TODO: revisar que se necesite

        print('Processing data ...')
        ts_data, static_data, meta_data = self.transform_data(y_df=y_df, X_s_df=X_s_df, X_t_df=X_t_df)

        print('Creating dataloader ...')
        ts_data = TimeseriesDataset(model='nbeats',
                                    ts_data=ts_data, static_data=static_data, meta_data=meta_data,
                                    window_sampling_limit=self.window_sampling_limit,
                                    offset=offset, input_size=self.input_size, output_size=self.output_size,
                                    idx_to_sample_freq=1,
                                    batch_size=self.batch_size)

        n_x_t, n_s_t = 0, 0
        if X_t_df is not None:
            n_x_t = X_t_df.shape[1]-2 # 2 for unique_id and ds
        if X_s_df is not None:
            n_s_t = X_s_df.shape[1]-1 # 1 for unique_id

        return ts_data, n_x_t, n_s_t

    def evaluate_performance(self, offset, losses):
        self.model.eval()
        loss_dict = {}
        with t.no_grad():
            y_hat, y, y_mask = self.predict(offset=offset, eval_mode=True)
            for loss_name in losses:
                loss_fn = self.__loss_fn(loss_name)
                loss = loss_fn(x=None, freq=self.seasonality, forecast=y_hat, target=y, mask=y_mask)
                loss_dict[loss_name] = loss.cpu().detach().numpy()
        return loss_dict

    def fit(self, y_df, X_s_df=None, X_t_df=None, offset=0, n_iterations=None, verbose=True, display_steps=100):
        # Random Seeds (model initialization)
        t.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # Parse data
        self.unique_ids = y_df['unique_id'].unique()
        self.ts_dataset, self.n_x_t, self.n_x_s = self.parse_data(y_df=y_df, X_s_df=X_s_df, X_t_df=X_t_df, offset=offset)

        # Instantiate model
        block_list = self.create_stack()
        self.model = NBeats(t.nn.ModuleList(block_list)).to(self.device)

        # Overwrite n_iterations and train datasets
        if n_iterations is None:
            n_iterations = self.n_iterations
        
        # Update offset (only for online train)
        self.ts_dataset.update_offset(offset)

        print('='*30+' Training NBEATS '+'='*30)

        dataloader = iter(self.ts_dataset)

        lr_decay_steps = n_iterations // self.n_lr_decay_steps
        if lr_decay_steps == 0:
            lr_decay_steps = 1

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_steps, gamma=self.lr_decay)
        training_loss_fn = self.__loss_fn(self.loss)
        
        self.loss_dict = {} # Restart self.loss_dict
        start = time.time()
        # Training Loop
        for step in range(n_iterations):
            self.model.train()
            self.ts_dataset.train()

            batch = next(dataloader)

            insample_y = self.to_tensor(batch['insample_y'])
            insample_x_t = self.to_tensor(batch['insample_x_t'])
            insample_mask = self.to_tensor(batch['insample_mask'])

            outsample_x_t = self.to_tensor(batch['outsample_x_t'])
            outsample_y = self.to_tensor(batch['outsample_y'])
            outsample_mask = self.to_tensor(batch['outsample_mask'])

            optimizer.zero_grad()
            forecast = self.model(insample_y, insample_x_t, insample_mask, outsample_x_t)

            training_loss = training_loss_fn(x=insample_y, freq=self.seasonality, forecast=forecast,
                                            target=outsample_y, mask=outsample_mask)

            if np.isnan(float(training_loss)):
                break

            training_loss.backward()
            t.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            lr_scheduler.step()
            if (verbose) & (step % display_steps == 0):                    
                print('Step:', '%d,' % step,
                      'Time: {:03.3f},'.format(time.time()-start),
                      '{} Loss: {:.9f},'.format(self.loss, training_loss.cpu().data.numpy()))

                if offset > 0:
                    loss = self.evaluate_performance(offset, self._eval_criterions)
                    print("Outsample loss: {}".format(loss))

                self.model.train()
                self.ts_dataset.train()
        
        # Last outsample prediction
        if offset > 0:
            self.loss_dict = self.evaluate_performance(offset, self._eval_criterions)
            print("Outsample loss: {}".format(self.loss_dict))

    def predict(self, X_test=None, offset=0, eval_mode=False):
        self.ts_dataset.update_offset(offset)
        self.ts_dataset.eval()

        # Build forecasts
        unique_ids = self.ts_dataset.get_meta_data_var('unique_id')
        last_ds = self.ts_dataset.get_meta_data_var('last_ds') #TODO: ajustar of offset
        
        batch = next(iter(self.ts_dataset))
        insample_y = self.to_tensor(batch['insample_y'])
        insample_x_t = self.to_tensor(batch['insample_x_t'])
        insample_mask = self.to_tensor(batch['insample_mask'])

        outsample_x_t = self.to_tensor(batch['outsample_x_t'])
        outsample_y = self.to_tensor(batch['outsample_y'])
        outsample_mask = self.to_tensor(batch['outsample_mask'])

        self.model.eval()
        with t.no_grad():
            forecast = self.model(insample_y, insample_x_t, insample_mask, outsample_x_t)

        if eval_mode:
            return forecast, outsample_y, outsample_mask

        # Predictions for panel
        Y_hat_panel = pd.DataFrame(columns=['unique_id', 'ds'])
        for i, unique_id in enumerate(unique_ids):
            Y_hat_id = pd.DataFrame([unique_id]*self.output_size, columns=["unique_id"])
            ds = pd.date_range(start=last_ds[i], periods=self.output_size+1, freq=self.frequency)
            Y_hat_id["ds"] = ds[1:]
            Y_hat_panel = Y_hat_panel.append(Y_hat_id, sort=False).reset_index(drop=True)

        forecast = forecast.cpu().detach().numpy()
        Y_hat_panel['y_hat'] = forecast.flatten()

        if X_test is not None:
            Y_hat_panel = X_test.merge(Y_hat_panel, on=['unique_id', 'ds'], how='left')
        
        return Y_hat_panel

    def predict_decomposed(self, X_test=None, offset=None):
        self.ts_dataset.update_offset(offset)
        self.ts_dataset.eval()

        # Build forecasts
        unique_ids = self.ts_dataset.get_meta_data_var('unique_id')
        last_ds = self.ts_dataset.get_meta_data_var('last_ds') #TODO: ajustar of offset

        batch = next(iter(dataloader))
        insample_y = self.to_tensor(batch['insample_y'])
        insample_x_t = self.to_tensor(batch['insample_x_t'])
        insample_mask = self.to_tensor(batch['insample_mask'])

        outsample_x_t = self.to_tensor(batch['outsample_x_t'])
        outsample_y = self.to_tensor(batch['outsample_y'])
        outsample_mask = self.to_tensor(batch['outsample_mask'])

        self.model.eval()
        with t.no_grad():
            forecast, forecast_blocks = self.model.decomposed_prediction(insample_y, insample_x_t, insample_mask, outsample_x_t)

        forecast = forecast.cpu().detach().numpy()
        decomposition = []
        for i in range(len(forecast_blocks)):
            decomposition.append(forecast_blocks[i].cpu().detach().numpy())
        
        return forecast, decomposition

    def save(self, model_dir, model_id):
    
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
        model_file = os.path.join(model_dir, f"model_{model_id}.model")
        print('Saving model to:\n {}'.format(model_file)+'\n')
        t.save({'model_state_dict': self.model.state_dict()}, model_file)

    def load(self, model_dir, model_id):

        model_file = os.path.join(model_dir, f"model_{model_id}.model")
        path = Path(model_file)

        assert path.is_file(), 'No model_*.model file found in this path!'

        print('Loading model from:\n {}'.format(model_file)+'\n')

        checkpoint = t.load(model_file, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)