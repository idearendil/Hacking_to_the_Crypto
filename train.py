import os
import glob
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

LOG_FOLDER = 'train1'
FEATURE_NUM = 139

# -----------------------------
# 1. Dataset 정의
# -----------------------------

class CryptoDataset(Dataset):
    def __init__(self, file_paths, seq_len=96, train=True, train_ratio=0.8, 
                 load=False, samples_path="temp.pkl"):
        """
        file_paths: CSV 파일 경로 리스트
        seq_len: LSTM 시퀀스 길이
        train: True면 train set, False면 val set
        train_ratio: train/val 비율
        load: True면 samples_path에서 pickle로 로드
        samples_path: pickle 파일 경로
        """
        self.seq_len = seq_len

        if load:
            with open(samples_path, "rb") as f:
                self.samples = pickle.load(f)
            print(f"Loaded {len(self.samples)} samples from {samples_path}")
        else:
            self.samples = []  # (file_path, start_idx) 튜플 저장
            for fp in tqdm(file_paths):
                df = pd.read_csv(fp)
                n_total = len(df)
                split_idx = int(n_total * train_ratio)

                if train:
                    idx_range = range(0, split_idx)
                else:
                    idx_range = range(split_idx, n_total)

                # sliding window index
                for i in range(len(idx_range) - seq_len):
                    self.samples.append((fp, idx_range[i]))

            with open(samples_path, "wb") as f:
                pickle.dump(self.samples, f)
            print(f"Saved {len(self.samples)} samples to {samples_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fp, start_idx = self.samples[idx]
        df = pd.read_csv(fp)
        x = df.drop(columns=['label']).values.astype(np.float32)
        y = df['label'].values.astype(np.float32)
        seq_x = x[start_idx:start_idx+self.seq_len]
        seq_y = y[start_idx+self.seq_len-1]  # 마지막 타임스텝의 label
        return torch.tensor(seq_x), torch.tensor(seq_y)
    
# -----------------------------
# 2. Selective LSTM 모델 정의
# -----------------------------
class SelectiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=False)
        self.fc_class = nn.Linear(hidden_size, 1)
        self.fc_gate = nn.Linear(hidden_size, 1)  # 선택 헤드

    def forward(self, x):
        # x: (B, seq_len, input_size)
        out, _ = self.lstm(x)  # out: (B, seq_len, hidden_size)
        h_last = out[:, -1, :]  # 마지막 타임스텝
        y_hat = torch.sigmoid(self.fc_class(h_last)).squeeze(1)  # (B,)
        s = torch.sigmoid(self.fc_gate(h_last)).squeeze(1)       # (B,)
        return y_hat, s

# -----------------------------
# 3. Selective Loss 정의
# -----------------------------
def selective_loss(y_hat, s, y, c0=0.5, lam=64.0, eps=1e-8):
    bce = nn.functional.binary_cross_entropy(y_hat, y, reduction='none')
    sel_risk = (s * bce).sum() / (s.sum() + eps)
    coverage = s.mean()
    penalty = torch.clamp(c0 - coverage, min=0.0) ** 2
    loss = sel_risk + lam * penalty
    return loss, sel_risk.detach(), coverage.detach()


if __name__ == "__main__":
    # -----------------------------
    # 4. 데이터 준비
    # -----------------------------

    folder = "preprocessed_data_min15_350000"
    all_files = sorted(glob.glob(os.path.join(folder, "*.csv")))

    # 데이터셋
    train_dataset = CryptoDataset(all_files, seq_len=96, train=True, train_ratio=0.8, load=True, samples_path="train_set.pkl")
    val_dataset   = CryptoDataset(all_files, seq_len=96, train=False, train_ratio=0.8, load=True, samples_path="val_set.pkl")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    input_size = FEATURE_NUM  # feature 수

    # -----------------------------
    # 5. 학습 준비
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SelectiveLSTM(input_size=input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 100

    # 학습 시작 전에 기록용 리스트 초기화
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_risk": [],
        "val_risk": [],
        "train_cov": [],
        "val_cov": []
    }

    csv_file = "training_history.csv"

    os.makedirs(f'{LOG_FOLDER}', exist_ok=True)
    best_val_risk = float('inf')  # 가장 낮은 val selective risk 기준

    # -----------------------------
    # 6. 학습 loop
    # -----------------------------
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_risk = 0
        total_cov = 0
        for x_batch, y_batch in tqdm(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_hat, s = model(x_batch)
            loss, sel_risk, coverage = selective_loss(y_hat, s, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            total_risk += sel_risk.item() * x_batch.size(0)
            total_cov += coverage.item() * x_batch.size(0)

        n_samples = len(train_dataset)
        train_loss_epoch = total_loss / n_samples
        train_risk_epoch = total_risk / n_samples
        train_cov_epoch = total_cov / n_samples

        # Validation
        model.eval()
        val_loss_total = 0
        val_risk_total = 0
        val_cov_total = 0
        with torch.no_grad():
            for x_batch, y_batch in tqdm(val_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_hat, s = model(x_batch)
                loss, sel_risk, coverage = selective_loss(y_hat, s, y_batch)
                val_loss_total += loss.item() * x_batch.size(0)
                val_risk_total += sel_risk.item() * x_batch.size(0)
                val_cov_total += coverage.item() * x_batch.size(0)

        n_val = len(val_dataset)
        val_loss_epoch = val_loss_total / n_val
        val_risk_epoch = val_risk_total / n_val
        val_cov_epoch = val_cov_total / n_val

        # 화면 출력
        print(f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss_epoch:.4f} | Train Risk: {train_risk_epoch:.4f} | Train Cov: {train_cov_epoch:.4f} | "
            f"Val Loss: {val_loss_epoch:.4f} | Val Risk: {val_risk_epoch:.4f} | Val Cov: {val_cov_epoch:.4f}")

        # -----------------------------
        # CSV 기록
        history["epoch"].append(epoch+1)
        history["train_loss"].append(train_loss_epoch)
        history["val_loss"].append(val_loss_epoch)
        history["train_risk"].append(train_risk_epoch)
        history["val_risk"].append(val_risk_epoch)
        history["train_cov"].append(train_cov_epoch)
        history["val_cov"].append(val_cov_epoch)

        if val_risk_epoch < best_val_risk:
            best_val_risk = val_risk_epoch
            torch.save(model.state_dict(), f"{LOG_FOLDER}/best.pth")
            print(f"Best model saved at epoch {epoch+1} with val_risk {best_val_risk:.4f}")

    # CSV 저장
    with open(f'{LOG_FOLDER}/{csv_file}', "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(history.keys())
        writer.writerows(zip(*history.values()))

    print("학습 완료")

    # 1) Loss 그래프
    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.savefig(f"{LOG_FOLDER}/loss_curve.png")
    plt.close()

    # 2) Selective Risk 그래프
    plt.figure()
    plt.plot(history["epoch"], history["train_risk"], label="train_risk")
    plt.plot(history["epoch"], history["val_risk"], label="val_risk")
    plt.xlabel("Epoch")
    plt.ylabel("Selective Risk")
    plt.title("Train vs Val Selective Risk")
    plt.legend()
    plt.savefig(f"{LOG_FOLDER}/selective_risk_curve.png")
    plt.close()

    # 3) Coverage 그래프
    plt.figure()
    plt.plot(history["epoch"], history["train_cov"], label="train_coverage")
    plt.plot(history["epoch"], history["val_cov"], label="val_coverage")
    plt.xlabel("Epoch")
    plt.ylabel("Coverage")
    plt.title("Train vs Val Coverage")
    plt.legend()
    plt.savefig(f"{LOG_FOLDER}/coverage_curve.png")
    plt.close()