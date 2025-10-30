from torch.utils.data import random_split
from dataset import SoundDS
import torch
import pandas as pd
import config

df_train = pd.read_csv(config.CSV_PATH_TRAIN)
ds_train = SoundDS(df=df_train, data_path=config.DATA_PATH_TRAIN, aug=True)

df_test = pd.read_csv(config.CSV_PATH_TEST)
ds_test = SoundDS(df=df_test, data_path=config.DATA_PATH_TEST, aug=False)

train_dl = torch.utils.data.DataLoader(ds_train, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE)
test_dl = torch.utils.data.DataLoader(ds_test, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE)