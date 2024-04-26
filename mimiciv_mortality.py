import gc

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
from multiprocessing import Manager
import pytorch_lightning as pl

class MIMICIVDataset(Dataset):
    def __init__(self, data_dir, split_name, n_timesteps=32, use_temp_cache=False, **kwargs):
        if split_name == 'val':
            self.data_dir = os.path.join(data_dir, 'train')
        else:
            self.data_dir = os.path.join(data_dir, split_name)
        self.split_name = split_name
        self.n_timesteps = n_timesteps
        self.temp_cache = Manager().dict() if use_temp_cache else None
        self.train_prop = 0.7
        self.val_prop = 0.15
        self.X = self.y = None
        self.predefined_categories = {
            #"Braden Activity": set(range(1, 5)),
            #"Braden Friction/Shear": set(range(1, 4)),
            #"Braden Mobility": set(range(1, 5)),
            #"Braden Moisture": set(range(1, 5)),
            #"Braden Nutrition": set(range(1, 5)),
            #"Braden Sensory Perception": set(range(1, 5)),
            "GCS - Eye Opening": set(range(1, 5)),
            "GCS - Motor Response": set(range(1, 7)),
            "GCS - Verbal Response": set(range(0, 6)),  # Including "No Response-ETT" as 0
            #"Goal Richmond-RAS Scale": set(range(-5, 1)),  # Negative and zero
            #"Pain Level": set(range(0, 9)),  # Includes "Unable to Score"
            #"Pain Level Response": set(range(0, 9)),  # Includes "Unable to Score"
            #"Richmond-RAS Scale": set(range(-5, 5)),  # Negative through positive
            #"Strength L Arm": set(range(0, 6)),
            #"Strength L Leg": set(range(0, 6)),
            #"Strength R Arm": set(range(0, 6)),
            #"Strength R Leg": set(range(0, 6)),
            #"Ambulatory aid": set(range(0, 8)),  # Including "Furniture" as 7
            "Capillary Refill L": set(range(1, 3)),
            "Capillary Refill R": set(range(1, 3)),
            #"Gait/Transferring": set(range(1, 6)),
            #"History of falling (within 3 mnths)": set(range(0, 2)),  # Yes or No
            #"IV/Saline lock": set(range(0, 2)),  # Yes or No
            #"Mental status": set(range(1, 3)),
            "Marital Status": set(range(1, 7)),  # Includes '' as 6
            "Insurance": set(range(1, 6)),  # Includes '' as 5
            "Admission Location": set(range(1, 14)),  # Includes '' as 13
            "Admission Type": set(range(1, 12)),  # Includes '' as 11
            "Ethnicity": set(range(0, 5)),  # Includes multiple ethnic groups and '' as 0
            "First Care Unit": set(range(1, 12))  # Includes '' as 11
        }


    def setup(self):
        # Load the list of stays
        listfile = os.path.join(self.data_dir, "listfile.csv")

        stay_list = pd.read_csv(listfile)
        # Randomly shuffle the DataFrame
        stay_list = stay_list.sample(frac=1, random_state=2020)  # Use a seed for reproducibility

        # Calculate split indices
        num_stays = len(stay_list)
        num_val = int(num_stays * self.val_prop / (self.val_prop + self.train_prop))  # First part for validation
        num_train = num_stays - num_val  # Rest part for training

        if self.split_name == 'val':
            # Use the first part for validation
            stay_list = stay_list.iloc[:num_val]
        elif self.split_name == 'train':
            # Use the rest part for training
            stay_list = stay_list.iloc[num_val:num_stays]
        else:
            # If split_name is not 'train' or 'val', no slicing is needed
            pass
        #stay_list = stay_list.iloc[:1000]

        timeseries_data = []
        labels = []

        # Load data for each stay
        for _, row in tqdm(stay_list.iterrows(), total=stay_list.shape[0], desc=f'Loading {self.split_name} data'):
            stay_id, label = row['stay'], row['y_true']
            ts_filename = os.path.join(self.data_dir, stay_id)

            # Read timeseries data
            ts_data = pd.read_csv(ts_filename)
            #ts_data = ts_data.iloc[:, :self.d_time_series_num()]
            #pd.set_option('display.max_rows', None)
            # print(ts_data.dtypes)
            ts_data.drop(columns=['Observation Window Length'], inplace=True)
            ts_data = ts_data.apply(pd.to_numeric, errors='coerce')
            for column, categories in self.predefined_categories.items():
                nan_category = -100
                ts_data[column] = ts_data[column].fillna(nan_category).astype(int)

                categories_with_nan = categories.union({nan_category})
                ts_data[column] = pd.Categorical(ts_data[column], categories=categories_with_nan)

                # Create dummy/one-hot encoded variables
                dummies = pd.get_dummies(ts_data[column], prefix=column)

                # Find the original column index
                col_index = ts_data.columns.get_loc(column)

                # Drop the original column
                ts_data.drop(columns=[column], inplace=True)

                # Concatenate data: part before the column, dummies, part after the column
                first_part = ts_data.iloc[:, :col_index]
                second_part = ts_data.iloc[:, col_index:]

                # Concatenate all parts together
                ts_data = pd.concat([first_part, dummies, second_part], axis=1)

            #pd.set_option('display.max_rows', None)
            #pd.set_option('display.max_columns', None)
            #for col in ts_data.columns:
            #    print(col)
            #print("col len=",len(ts_data.columns))
            #exit(1)
            # Store data
            timeseries_data.append(torch.tensor(ts_data.values, dtype=torch.float32))
            labels.append(label)

        max_length = 1250
        preprocessed_data = []
        min_padding_length = 1250
        # Pad sequences with NaN and store them
        for ts_data in timeseries_data:
            # Calculate how much padding is needed
            padding_length = max_length - ts_data.shape[0]
            if padding_length < min_padding_length:
                min_padding_length = padding_length
            # Pad the sequence with NaNs if necessary
            if padding_length > 0:
                padding = torch.full((padding_length, ts_data.shape[1]), float('nan'), dtype=torch.float32)
                ts_data_padded = torch.cat((ts_data, padding), dim=0)
            else:
                ts_data_padded = ts_data

            preprocessed_data.append(ts_data_padded)
        print(f'max_length = 1250, min_padding_length={min_padding_length}')
        # self.X = torch.stack(timeseries_data)
        self.X = torch.stack(preprocessed_data)
        self.y = torch.tensor(labels, dtype=torch.long).unsqueeze(1)

        self.means = []
        self.stds = []
        self.maxes = []
        self.mins = []
        for i in range(self.X.shape[2]):
            vals = self.X[:,:,i].flatten()
            vals = vals[~torch.isnan(vals)]
            if vals.numel() > 0:
                self.means.append(vals.mean())
                self.stds.append(vals.std())
                self.maxes.append(vals.max())
                self.mins.append(vals.min())
            else:
                self.means.append(0)
                self.stds.append(1)
                self.maxes.append(float('nan'))
                self.mins.append(float('nan'))
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if self.temp_cache is not None and i in self.temp_cache:
            return self.temp_cache[i]

        ins = self.X[i, ~torch.isnan(self.X[i, :, 0]), :]
        time = ins[:, 0] / 24  # input is HOURS
        x_static = torch.zeros(self.d_static_num())

        x_ts = torch.zeros((self.n_timesteps, self.d_time_series_num() * 2))
        for i_t, t in enumerate(time):
            bin = self.n_timesteps - 1 if t == time[-1] else int(t / time[-1] * self.n_timesteps)
            for i_ts in range(1, self.d_time_series_num()+1):
                x_i = ins[i_t, i_ts]
                if not torch.isnan(x_i).item():
                    x_ts[bin, i_ts - 1] = (x_i - self.means[i_ts]) / (self.stds[i_ts] + 1e-7)
                    x_ts[bin, i_ts - 1 + self.d_time_series_num()] += 1
        bin_ends = torch.arange(1, self.n_timesteps + 1) / self.n_timesteps * time[-1]

        for i_tab in range(self.d_time_series_num()+1, self.d_time_series_num()+self.d_static_num()+1):
            x_i = ins[0, i_tab]
            x_i = (x_i - self.means[i_tab]) / (self.stds[i_tab] + 1e-7)
            x_static[i_tab - self.d_time_series_num()-1] = x_i.nan_to_num(0.)

        x = (x_ts, x_static, bin_ends)
        y = self.y[i, 0]
        if self.temp_cache is not None:
            self.temp_cache[i] = (x, y)

        return x, y

    def d_static_num(self):
        return 60

    def d_time_series_num(self):
        return 159

    def d_target(self):
        return 1

    def pos_frac(self):
        return torch.mean(self.y.float()).item()

def collate_into_seqs(batch):
    xs, ys = zip(*batch)
    return zip(*xs), ys
class MIMICIVDataModule(pl.LightningDataModule):
    def __init__(self, data_path='./mimic-iv-benchmarks/data/in-hospital-mortality/', use_temp_cache=False, batch_size=8, num_workers=1, prefetch_factor=2,
            verbose=0, **kwargs):
        self.use_temp_cache = use_temp_cache
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self.data_path = data_path
        self.ds_train = MIMICIVDataset(self.data_path,'train', use_temp_cache=use_temp_cache)
        self.ds_val = MIMICIVDataset(self.data_path,'val', use_temp_cache=use_temp_cache)
        self.ds_test = MIMICIVDataset(self.data_path,'test', use_temp_cache=use_temp_cache)

        self.prepare_data_per_node = False

        self.dl_args = {'batch_size': self.batch_size, 'prefetch_factor': self.prefetch_factor,
                        'collate_fn': collate_into_seqs, 'num_workers': num_workers}

    def setup(self, stage=None):
        if stage is None:
            self.ds_train.setup()
            self.ds_val.setup()
            self.ds_test.setup()
        elif stage == 'fit':
            self.ds_train.setup()
            self.ds_val.setup()
        elif stage == 'validate':
            self.ds_val.setup()
        elif stage == 'test':
            self.ds_test.setup()
    def prepare_data(self):
        pass

    def _log_hyperparams(self):
        pass

    def train_dataloader(self):
        return DataLoader(self.ds_train, shuffle=True, **self.dl_args)

    def val_dataloader(self):
        return DataLoader(self.ds_val, **self.dl_args)

    def test_dataloader(self):
        return DataLoader(self.ds_test, **self.dl_args)

    def d_static_num(self):
        return self.ds_train.d_static_num()

    def d_time_series_num(self):
        return self.ds_train.d_time_series_num()

    def d_target(self):
        return self.ds_train.d_target()

    def pos_frac(self):
        return self.ds_train.pos_frac()
