import os
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

INPUT_DIR = "G:/preprocessed_data_hour4_21900"  # 전처리 완료된 CSV 폴더
LABEL = ["label1", "label2", "label3"]
BUILD_DATASET = True
TEST_RATIO = 0.2

if __name__ == "__main__":
    # --------------------------------
    # 1. 모든 CSV 불러와서 합치기
    # --------------------------------
    if BUILD_DATASET:
        for model_id in range(1, 4):
            train_list = []
            test_list = []
            files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]

            for f in files:
                path = os.path.join(INPUT_DIR, f)
                df = pd.read_csv(path)
                
                # label 없는 행 제외
                df = df.dropna(subset=[LABEL[0]])

                if model_id > 1:
                    df_pos = df[df[f"label{model_id-1}"] == 1]
                    df_neg = df[df[f"label{model_id-1}"] == 0].sample(frac=len(df_pos) / len(df) * 0.5, random_state=42)
                    df = pd.concat([df_pos, df_neg]).reset_index(drop=True)
                    df = df.sort_values(by='date', ascending=True)
                
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
            drop_lst = []
            for idx in range(1, 4):
                if idx != model_id:
                    drop_lst.append(f"label{idx}")
            train_df.drop(drop_lst, axis=1, inplace=True)
            test_df.drop(drop_lst, axis=1, inplace=True)
            train_df.to_csv(f"automl_data_label{model_id}_train.csv", index=False)
            test_df.to_csv(f"automl_data_label{model_id}_test.csv", index=False)

    # --------------------------------
    # 3. AutoGluon 학습
    # --------------------------------
    for model_id in range(2, 4):
        train_df = pd.read_csv(f"automl_data_label{model_id}_train.csv")
        test_df = pd.read_csv(f"automl_data_label{model_id}_test.csv")
        print(f"train: {len(train_df)}, test: {len(test_df)}")

        predictor = TabularPredictor(
            label=LABEL[model_id-1],
            problem_type="binary",
            eval_metric="accuracy",
            path=f"hour4_models_{model_id}"
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
        y_true = test_df[LABEL[model_id-1]]
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
        plt.savefig(f"automl2_pr_curve_{model_id}.png", dpi=300)
        plt.close()
