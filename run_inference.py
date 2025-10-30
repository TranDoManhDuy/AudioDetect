from collections import Counter
import config
from model import AudioClassification
from AudioUtil import AudioUtil, pipe_normAudio, audio2inputmodel
import torchaudio
import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import numpy as np

def sliding_windows(waveform, sample_rate, window_ms=200, hop_ms = 50, channels = 2):
    window_len = int(sample_rate * window_ms / 1000)
    hop_len = int(sample_rate * hop_ms / 1000)
    total_len = waveform.shape[1]
    
    windows = []
    start = 0
    
    while start < total_len:
        end = start + window_len
        segment = waveform[:, start:end]
        if segment.shape[1] < window_len:
            pad_len = window_len - segment.shape[1]
            pad = torch.zeros((channels, pad_len))
            segment = torch.cat((segment, pad), dim=1)
        
        windows.append(segment)
        start += hop_len
    return torch.stack(windows)

def visualization(waveform_windows, sr):
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=64
    )

    num_windows = waveform_windows.shape[0]
    num_channels = waveform_windows.shape[1]

    fig, axes = plt.subplots(num_windows, num_channels, figsize=(num_channels * 4, num_windows * 2))
    if num_windows == 1 and num_channels == 1:
        axes = [[axes]]
    elif num_windows == 1:
        axes = [axes]
    elif num_channels == 1:
        axes = [[ax] for ax in axes]

    for i in range(num_windows):
        for ch in range(num_channels):
            S = mel_spec(waveform_windows[i, ch])
            S_dB = torchaudio.functional.amplitude_to_DB(S, multiplier=10, amin=1e-10, db_multiplier=0)
            axes[i][ch].imshow(S_dB.numpy(), origin="lower", aspect="auto", cmap="magma")
            axes[i][ch].set_title(f"Win {i+1}, Ch {ch+1}")
            axes[i][ch].axis("off")

    plt.tight_layout()
    plt.show()

def plot_predictions_over_time(predictions):
    """
    Vẽ biểu đồ đường biểu diễn các giá trị dự đoán (0, 1, 2) qua từng window.
    """
    if not predictions:
        print("Không có dữ liệu dự đoán để vẽ.")
        return
        
    plt.figure(figsize=(15, 5))
    
    # Sử dụng plt.step để biểu đồ trông rõ ràng hơn cho các giá trị rời rạc
    plt.step(range(len(predictions)), predictions, where='mid', label='Predicted Class', marker='o', markersize=4)

    plt.title('Biểu đồ dự đoán của Model theo thời gian')
    plt.xlabel('Thứ tự Window (Time Step)')
    plt.ylabel('Lớp dự đoán (0, 1, hoặc 2)')
    
    # Đảm bảo trục Y chỉ hiển thị các giá trị 0, 1, 2
    plt.yticks([0, 1, 2])
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
# phần load model
run_model = torch.nn.DataParallel(AudioClassification())
run_model = run_model.to(config.DEVICE)

try:
    state_dict = torch.load("train_rs.pt", map_location=config.DEVICE)
    run_model.load_state_dict(state_dict)
    print("Model weights loaded successfully!")
except FileNotFoundError:
    print("Error: Saved model file 'train_rs.pt' not found. Please train the model first.")
    exit() # Thoát nếu không tìm thấy file
    
def demo(window_ms = 200, hop_ms = 50, path = ""):
    waveform, sr = AudioUtil.open(path)
    waveform, sr = pipe_normAudio((waveform, sr)) # chuẩn hóa channels và sample_rate
    waveform_windows = sliding_windows(waveform=waveform, sample_rate=sr, window_ms=window_ms, hop_ms=hop_ms) # cắt ra thành từng khúc
    # waveform_windows có kích thước: num_window, 2, 8820
    lsrs = []
    couter_rs = None
    with torch.inference_mode():
        for i in range(waveform_windows.shape[0]):
            inputmodel = audio2inputmodel((waveform_windows[i], sr))
            _, result = torch.max(run_model(inputmodel), 1, keepdim=True)
            lsrs.append(*result.reshape(-1).tolist())
        couter_rs = Counter(lsrs)
    return couter_rs, lsrs

def find_window():
    lsCounter_0 = []
    lsCounter_1 = []
    lsCounter_2 = []

    for i in range(200, 2000):
        rs, _ = demo(window_ms=i, hop_ms=50)
        lsCounter_0.append(rs[0])
        lsCounter_1.append(rs[1])
        lsCounter_2.append(rs[2])

    x = np.arange(len(lsCounter_2))
    # Vẽ biểu đồ đường
    plt.figure(figsize=(10, 5))
    plt.plot(x, lsCounter_0, label="Class: No", linewidth=2)
    plt.plot(x, lsCounter_1, label="Class: Yes", linewidth=2)
    plt.plot(x, lsCounter_2, label="Class: other", linewidth=2)

    plt.title("Biểu đồ đường")
    plt.xlabel("Chỉ số phần tử")
    plt.ylabel("Giá trị")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("sliding_window_rs.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__=="__main__":
    counter, lsrs = demo(window_ms=2000, hop_ms=50, path=r"D:\ptithcm\HTTM\Data outline\data\bird\0a9f9af7_nohash_0.wav")
    print(counter, lsrs)
    time = np.arange(len(lsrs))
    plt.figure(figsize=(10, 3))
    plt.step(time, lsrs, where='mid', linewidth=2)
    plt.yticks([0, 1, 2], ['class 0', 'class 1', 'class 2'])
    plt.xlabel("Thời gian (frame)")
    plt.ylabel("Nhãn")
    plt.title("Biểu đồ nhãn theo thời gian (lsrs)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()