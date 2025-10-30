from torch.utils.data import Dataset
import torchaudio
from AudioUtil import AudioUtil
import os

class SoundDS(Dataset):
    def __init__(self, df, data_path, aug = True):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 2000
        self.sr = 44100
        self.channels = 2
        self.shift_pct = 0.4
        self.aug = aug
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index] # lấy dòng dữ liệu trong dataframe
        audio_path = row["filename"]
        if row["label"] == "yes":
            class_id = 1
        elif row["label"] == "no":
            class_id = 0
        else:
            class_id = 2
        aud = AudioUtil.open(audio_path=audio_path)
        reaud = AudioUtil.resample(aud=aud, newsr=self.sr)
        rechan = AudioUtil.rechannel(aud=reaud, new_channel=self.channels)
        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        sgram = AudioUtil.spectrogram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        if self.aug:
            sgram = AudioUtil.spectrogram_augment(sgram, max_mask_pct=0.1, n_freq_mask=1, n_time_masks=1)
        return sgram, class_id