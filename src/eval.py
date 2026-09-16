import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, ElectraForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def ensemble_predict(dataloader, model_dir_prefix, num_folds, device):
    """
    K-Fold로 학습된 여러 모델을 차례로 불러와 예측 확률(Logits)을 합산하는 함수
    """
    num_samples = len(dataloader.dataset)
    num_classes = 4 
    
    ensemble_logits = torch.zeros((num_samples, num_classes)).to(device)
    
    true_labels = np.array(dataloader.dataset.labels)
    
    for fold_idx in range(1, num_folds + 1):
        fold_path = f"{model_dir_prefix}{fold_idx}"
        
        if not os.path.exists(fold_path):
            print(f"경고: '{fold_path}' 경로를 찾을 수 없습니다. 건너뜁니다.")
            continue
            
        print(f"\n--- [Fold {fold_idx}] 모델 로드 및 예측 중 ---")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(fold_path)
            model = ElectraForSequenceClassification.from_pretrained(fold_path, num_labels=4, use_safetensors=True)
            model.to(device)
            model.eval()
        except Exception as e:
            print(f" 모델 로드 실패: {e}")
            continue

        fold_preds = []
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for batch in tqdm(dataloader, desc=f"Fold {fold_idx} Inference", leave=False):
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                fold_preds.append(outputs.logits)
        
        fold_tensor = torch.cat(fold_preds, dim=0)
        ensemble_logits += fold_tensor
        
        del model
        torch.cuda.empty_cache()
    
    return ensemble_logits, true_labels

if __name__ == "__main__":
    TEST_CSV_PATH = './data/final/test_final_holdout.csv' 
    MODEL_DIR_PREFIX = './artifacts/saved_models/best_sentiment_model_fold' 
    SAVE_RESULT_PATH = './artifacts/ensemble_predictions_result.csv'
    
    BATCH_SIZE = 32
    NUM_FOLDS = 5
    PRETRAINED_MODEL_NAME = "beomi/KcELECTRA-base-v2022"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    if not os.path.exists(TEST_CSV_PATH):
        print(f" 오류: '{TEST_CSV_PATH}' 파일을 찾을 수 없습니다.")
        exit()
        
    print(f"'{TEST_CSV_PATH}'에서 최종 테스트 데이터를 불러옵니다...")
    test_df = pd.read_csv(TEST_CSV_PATH)
    
    test_df.dropna(subset=['processed_review', 'sentiment_label'], inplace=True)
    test_df['sentiment_label'] = test_df['sentiment_label'].astype(int)

    test_texts = test_df['processed_review'].tolist()
    test_labels = test_df['sentiment_label'].tolist()
    print(f"테스트 데이터 개수: {len(test_texts)}개")

    print("토크나이저 로드 및 데이터 변환 중...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_DIR_PREFIX}1")
    except:
        print("저장된 토크나이저를 못 찾아 원본(HuggingFace)에서 로드합니다.")
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)
    test_dataset = ReviewDataset(test_encodings, test_labels)
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print("\n5-Fold 앙상블 예측 시작...")
    logits, true_labels = ensemble_predict(test_loader, MODEL_DIR_PREFIX, NUM_FOLDS, device)

    final_preds = torch.argmax(logits, dim=1).cpu().numpy()

    test_df['predicted_label'] = final_preds
    os.makedirs(os.path.dirname(SAVE_RESULT_PATH), exist_ok=True)
    test_df.to_csv(SAVE_RESULT_PATH, index=False, encoding='utf-8-sig')
    print(f"\n예측 결과가 '{SAVE_RESULT_PATH}'에 저장되었습니다.")

    print("\n" + "="*40)
    print("         최종 테스트 결과 (Ensemble)        ")
    print("="*40)

    acc = accuracy_score(true_labels, final_preds)
    print(f"\n최종 정확도 (Accuracy): {acc:.4f}\n")

    target_names = ['매우 부정(1)', '부정(2)', '긍정(4)', '매우 긍정(5)']
    print(classification_report(true_labels, final_preds, target_names=target_names, digits=4))

    print("\n혼동 행렬 시각화 출력 중...")
    cm = confusion_matrix(true_labels, final_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix (5-Fold Ensemble)', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.show()
