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
                 load=False, samples_path="temp.pkl", processed_folder="processed_blocks"):
        """
        file_paths: CSV 파일 경로 리스트
        seq_len: LSTM 시퀀스 길이
        train: True면 train set, False면 val set
        train_ratio: train/val 비율
        load: True면 samples_path에서 pickle로 로드
        samples_path: pickle 파일 경로 (sample index 리스트 저장)
        processed_folder: seq_len 단위로 분할한 데이터 블록 저장 폴더 (겹치지 않게)
        """
        self.seq_len = seq_len
        os.makedirs(processed_folder, exist_ok=True)
        self.processed_folder = processed_folder

        if load:
            with open(samples_path, "rb") as f:
                self.samples = pickle.load(f)
            print(f"Loaded {len(self.samples)} samples from {samples_path}")
        else:
            self.samples = []  # (block_file, local_start_idx) 저장
            for fp in tqdm(file_paths):
                df = pd.read_csv(fp)
                n_total = len(df)
                split_idx = int(n_total * train_ratio)
                if train:
                    idx_range = range(0, split_idx)
                else:
                    idx_range = range(split_idx, n_total)

                # 1) 겹치지 않게 block 단위 저장
                block_id = 0
                for start in range(idx_range.start, idx_range.stop, seq_len):
                    end = min(start + seq_len, idx_range.stop)
                    block_df = df.iloc[start:end]
                    x_tensor = torch.tensor(block_df.drop(columns=['label']).values, dtype=torch.float32)
                    y_tensor = torch.tensor(block_df['label'].values, dtype=torch.float32)
                    block = (x_tensor, y_tensor)  # block 단위
                    block_file = os.path.join(processed_folder, f"{os.path.basename(fp)}_block{block_id}.pkl")
                    with open(block_file, "wb") as f:
                        pickle.dump(block, f)
                    block_id += 1

                # 2) 슬라이딩 윈도우 인덱스 저장
                total_len = len(idx_range)
                for i in range(total_len - seq_len):
                    # 어떤 block에 속하는지 찾기
                    block_idx = i // seq_len
                    local_start = i % seq_len
                    block_file = os.path.join(processed_folder, f"{os.path.basename(fp)}_block{block_idx}.pkl")
                    self.samples.append((block_file, local_start))

            # sample list 저장
            with open(samples_path, "wb") as f:
                pickle.dump(self.samples, f)
            print(f"Saved {len(self.samples)} samples to {samples_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        block_file, local_start = self.samples[idx]

        # 현재 block 로드
        with open(block_file, "rb") as f:
            x_block, y_block = pickle.load(f)

        # 만약 필요한 길이가 block을 넘어가면 → 다음 block도 불러오기
        if local_start + self.seq_len > len(x_block):
            # block 이름에서 block index 추출
            base, block_name = os.path.split(block_file)
            prefix, block_id = block_name.rsplit("_block", 1)
            next_block_file = os.path.join(base, f"{prefix}_block{int(block_id[:-4])+1}.pkl")

            with open(next_block_file, "rb") as f:
                x_next, y_next = pickle.load(f)

            # 두 block 이어붙이기
            x_block = torch.cat([x_block, x_next], dim=0)
            y_block = torch.cat([y_block, y_next], dim=0)

        # 이제 슬라이싱
        x = x_block[local_start:local_start+self.seq_len]
        y = y_block[local_start+self.seq_len-1]
        return x, y
        
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
        "val_cov": [],
        "val_acc_s03": [],
        "val_acc_s05": [],
        "val_acc_s07": []
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

        # 정확도 계산용 변수
        correct_s03 = 0; total_s03 = 0
        correct_s05 = 0; total_s05 = 0
        correct_s07 = 0; total_s07 = 0

        with torch.no_grad():
            for x_batch, y_batch in tqdm(val_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                y_hat, s = model(x_batch)
                loss, sel_risk, coverage = selective_loss(y_hat, s, y_batch)

                val_loss_total += loss.item() * x_batch.size(0)
                val_risk_total += sel_risk.item() * x_batch.size(0)
                val_cov_total += coverage.item() * x_batch.size(0)

                # -----------------------------
                # coverage threshold별 정확도
                # -----------------------------
                y_pred = (y_hat > 0.5).float()

                mask_s03 = (s > 0.3)
                if mask_s03.any():
                    correct_s03 += (y_pred[mask_s03] == y_batch[mask_s03]).sum().item()
                    total_s03 += mask_s03.sum().item()

                mask_s05 = (s > 0.5)
                if mask_s05.any():
                    correct_s05 += (y_pred[mask_s05] == y_batch[mask_s05]).sum().item()
                    total_s05 += mask_s05.sum().item()

                mask_s07 = (s > 0.7)
                if mask_s07.any():
                    correct_s07 += (y_pred[mask_s07] == y_batch[mask_s07]).sum().item()
                    total_s07 += mask_s07.sum().item()

        n_val = len(val_dataset)
        val_loss_epoch = val_loss_total / n_val
        val_risk_epoch = val_risk_total / n_val
        val_cov_epoch = val_cov_total / n_val

        # threshold별 정확도 계산
        val_acc_s03 = correct_s03 / total_s03 if total_s03 > 0 else 0
        val_acc_s05 = correct_s05 / total_s05 if total_s05 > 0 else 0
        val_acc_s07 = correct_s07 / total_s07 if total_s07 > 0 else 0

        # 화면 출력
        print(f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss_epoch:.4f} | Train Risk: {train_risk_epoch:.4f} | Train Cov: {train_cov_epoch:.4f} | "
            f"Val Loss: {val_loss_epoch:.4f} | Val Risk: {val_risk_epoch:.4f} | Val Cov: {val_cov_epoch:.4f}")

        # -----------------------------
        # CSV 기록
        # -----------------------------
        history["epoch"].append(epoch+1)
        history["train_loss"].append(train_loss_epoch)
        history["val_loss"].append(val_loss_epoch)
        history["train_risk"].append(train_risk_epoch)
        history["val_risk"].append(val_risk_epoch)
        history["train_cov"].append(train_cov_epoch)
        history["val_cov"].append(val_cov_epoch)
        history["val_acc_s03"].append(val_acc_s03)
        history["val_acc_s05"].append(val_acc_s05)
        history["val_acc_s07"].append(val_acc_s07)


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

    # Accuracy curves (s thresholds)
    plt.figure()
    plt.plot(history["epoch"], history["val_acc_s03"], label="s>0.3")
    plt.plot(history["epoch"], history["val_acc_s05"], label="s>0.5")
    plt.plot(history["epoch"], history["val_acc_s07"], label="s>0.7")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy at different s thresholds")
    plt.legend()
    plt.savefig(f"{LOG_FOLDER}/val_accuracy_curve.png")
    plt.close()