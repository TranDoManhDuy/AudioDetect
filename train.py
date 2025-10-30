import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import config
import torch.nn.functional as F 
from dataloader import train_dl, test_dl
from model import AudioClassification
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

print(torch.cuda.get_device_name(0))
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

# # ========================================================= testing
def testing(model, test_dl):
    model.eval()
    running_loss = 0
    correct_prediction = 0
    total_prediction = 0
    criterion = nn.CrossEntropyLoss() 
    all_predictions = []
    all_labels = []

    # 2. Bắt đầu context torch.no_grad()
    with torch.inference_mode():
        test_loader_tqdm = tqdm(test_dl, desc="Testing")
        
        for i, data in enumerate(test_loader_tqdm):
            inputs, labels = data[0].to(config.DEVICE), data[1].to(config.DEVICE)
            
            target_size = inputs.shape[2]
            inputs = F.interpolate(
                inputs, 
                size=(target_size, target_size), 
                mode='bilinear', 
                align_corners=False
            )
            inputs = (inputs - inputs.mean(dim=(1,2,3), keepdim=True)) / (inputs.std(dim=(1,2,3), keepdim=True) + 1e-6)
            
            # Đưa dữ liệu qua model
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Tính toán các chỉ số
            running_loss += loss.item()
            _, prediction = torch.max(outputs, 1)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
            
            # Lưu predictions và labels
            all_predictions.extend(prediction.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Cập nhật thanh tiến trình
            avg_loss_process = running_loss / (i + 1)
            acc_process = correct_prediction / total_prediction
            test_loader_tqdm.set_postfix(loss=f"{avg_loss_process:.4f}", acc=f"{acc_process:.4f}")

    # In kết quả cuối cùng
    avg_loss = running_loss / len(test_dl)
    avg_acc = correct_prediction / total_prediction
    print(f"\nTest Results: Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
    
    # ==================== CONFUSION MATRIX ====================
    # Tạo confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Hiển thị confusion matrix dạng heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Test Set')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    # In classification report
    print("\n" + "="*50)
    print("Classification Report:")
    print("="*50)
    print(classification_report(all_labels, all_predictions))
    print("Confusion Matrix:")
    print(cm)
    return avg_acc

def exam(model, test_dl, writer, epoch):
    running_loss = 0
    correct_prediction = 0
    total_prediction = 0
    criterion = nn.CrossEntropyLoss() 
    all_predictions = []
    all_labels = []

    test_loader_tqdm = tqdm(test_dl, desc="Exam")
    
    for i, data in enumerate(test_loader_tqdm):
        inputs, labels = data[0].to(config.DEVICE), data[1].to(config.DEVICE)
        
        target_size = inputs.shape[2]
        inputs = F.interpolate(
            inputs, 
            size=(target_size, target_size), 
            mode='bilinear', 
            align_corners=False
        )
        inputs = (inputs - inputs.mean(dim=(1,2,3), keepdim=True)) / (inputs.std(dim=(1,2,3), keepdim=True) + 1e-6)
        
        # Đưa dữ liệu qua model
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Tính toán các chỉ số
        running_loss += loss.item()
        _, prediction = torch.max(outputs, 1)
        correct_prediction += (prediction == labels).sum().item()
        total_prediction += prediction.shape[0]
        
        # Lưu predictions và labels
        all_predictions.extend(prediction.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Cập nhật thanh tiến trình
        avg_loss_process = running_loss / (i + 1)
        acc_process = correct_prediction / total_prediction
        test_loader_tqdm.set_postfix(loss=f"{avg_loss_process:.4f}", acc=f"{acc_process:.4f}")

    # In kết quả cuối cùng
    avg_loss = running_loss / len(test_dl)
    avg_acc = correct_prediction / total_prediction
    writer.add_scalar("Loss/exam", avg_loss, epoch)
    writer.add_scalar("Acc/exam", avg_acc, epoch)
    print(f"\nTest Results: Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

def training(model, train_dl, num_epochs):
    writer = SummaryWriter()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy="linear")
    # repeat for each epoch
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        correct_prediction = 0
        total_prediction = 0
        train_loader_tqdm = tqdm(train_dl, desc=f"Epoch {epoch + 1} / {num_epochs}")
        
        for i, data in enumerate(train_loader_tqdm):
            inputs, labels = data[0].to(config.DEVICE), data[1].to(config.DEVICE)
            target_size = inputs.shape[2]
            inputs = F.interpolate(
                inputs,
                size=(target_size, target_size), 
                mode='bilinear', 
                align_corners=False
            )
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            _, prediction = torch.max(outputs, 1)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            avg_loss_process = running_loss / (i + 1)
            acc_process = correct_prediction / total_prediction
            train_loader_tqdm.set_postfix(loss=f"{avg_loss_process:.4f}", acc=f"{acc_process:.4f}")
            
        num_batchs = len(train_dl)
        avg_loss = running_loss / num_batchs
        avg_acc = correct_prediction/total_prediction
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Acc/train", avg_acc, epoch)
        print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.2f}, Accuracy: {avg_acc:.2f}")
        print("\n" + "=" * 50 + "\n Start Exam")
        exam(model, test_dl, writer, epoch)
        print()
    torch.save(model.state_dict(), "train_rs.pt")
    print("Finished Training")

num_epochs = 50
model = torch.nn.DataParallel(AudioClassification())
model = model.to(config.DEVICE)
next(model.parameters()).device
training(model, train_dl, num_epochs)

# 1. Khởi tạo lại kiến trúc model
# QUAN TRỌNG: Khởi tạo model y hệt như lúc train
test_model = torch.nn.DataParallel(AudioClassification())
test_model = test_model.to(config.DEVICE)

# 2. Tải các trọng số đã được huấn luyện
# Đảm bảo file "train_rs.pt" nằm đúng đường dẫn
try:
    state_dict = torch.load("train_rs.pt", map_location=config.DEVICE)
    test_model.load_state_dict(state_dict)
    print("Model weights loaded successfully!")
except FileNotFoundError:
    print("Error: Saved model file 'train_rs.pt' not found. Please train the model first.")
    exit() # Thoát nếu không tìm thấy file

# 3. Gọi hàm testing
# Giả sử bạn đã có 'test_dl'
print("\nStarting evaluation on the test set...")
testing(test_model, test_dl)