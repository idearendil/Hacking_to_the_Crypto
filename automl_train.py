import os
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

INPUT_DIR = "G:/preprocessed_data_day_3500"  # 전처리 완료된 CSV 폴더
LABEL = "label"
LOAD_DATASET = True
TEST_RATIO = 0.2

# --------------------------------
# 1. 모든 CSV 불러와서 합치기
# --------------------------------
if not LOAD_DATASET:
    train_list = []
    test_list = []
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]

    for f in files:
        path = os.path.join(INPUT_DIR, f)
        df = pd.read_csv(path)
        
        # label 없는 행 제외
        df = df.dropna(subset=[LABEL])
        
        # 어떤 코인 데이터인지 구분하는 컬럼 추가 (모델에 유용할 수 있음)
        df["coin"] = f.replace(".csv", "")
        
        if len(df) > 5:  # 최소 길이 확보
            split_idx = int(len(df) * (1 - TEST_RATIO))
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
        else:
            # 데이터가 너무 짧으면 전부 train으로
            train_df = df
            test_df = pd.DataFrame(columns=df.columns)
        
        train_list.append(train_df)
        test_list.append(test_df)

    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    train_df.to_csv("automl_data_day_3500_train.csv", index=False)
    test_df.to_csv("automl_data_day_3500_test.csv", index=False)    
else:
    train_df = pd.read_csv("automl_data_day_3500_train.csv")
    test_df = pd.read_csv("automl_data_day_3500_test.csv")

# --------------------------------
# 3. AutoGluon 학습
# --------------------------------
predictor = TabularPredictor(
    label=LABEL,
    problem_type="binary",
    eval_metric="balanced_accuracy",
    path="ag_models_balanced"
).fit(
    train_df,
    presets="extreme_quality",
)

# --------------------------------
# 4. 평가 & 결과 확인
# --------------------------------
performance = predictor.evaluate(test_df)
print("성능:", performance)

# 리더보드
leaderboard = predictor.leaderboard(test_df, silent=True)
print(leaderboard)

# --------------------------------
# 4-1. Precision-Recall Curve 저장
# --------------------------------
y_true = test_df[LABEL]
y_proba = predictor.predict_proba(test_df)[1]  # positive 클래스 확률 (label=1일 때)

precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
avg_precision = average_precision_score(y_true, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"AP={avg_precision:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("precision_recall_curve.png", dpi=300)
plt.close()

# --------------------------------
# 5. 예측 사용 예시
# --------------------------------
preds = predictor.predict(test_df.head(10))
print(preds)

proba = predictor.predict_proba(test_df.head(10))
print(proba)