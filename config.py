import torch

BATCH_SIZE = 8
CSV_PATH_TRAIN = r"D:\ptithcm\HTTM\Driver drowsiness audio\dataset\dataset_train_aug2.csv"
DATA_PATH_TRAIN = r"D:\ptithcm\HTTM\Driver drowsiness audio\dataset\train_aug2"
CSV_PATH_TEST = r"D:\ptithcm\HTTM\Driver drowsiness audio\dataset\dataset_test_aug2.csv"
DATA_PATH_TEST = r"D:\ptithcm\HTTM\Driver drowsiness audio\dataset\test_aug2"
SHUFFLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")