import pandas as pd
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from transformers import ElectraForSequenceClassification
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

def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy

if __name__ == "__main__":
    BATCH_SIZE = 32 
    N_FOLDS = 5
    EPOCHS_PER_FOLD = 5
    LEARNING_RATE = 5e-5
    
    PRETRAINED_MODEL_NAME = "beomi/KcELECTRA-base-v2022"
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    print(f"사용 디바이스: {DEVICE}")

    data_dir = './data/final'
    kfold_data_path = os.path.join(data_dir, 'train_for_kfold.csv')
    holdout_data_path = os.path.join(data_dir, 'test_final_holdout.csv')
    save_dir_base = './artifacts/saved_models'

    df_cv = pd.read_csv(kfold_data_path)
    df_test = pd.read_csv(holdout_data_path)

    print("결측치 제거 및 데이터 정리")
    df_cv.dropna(subset=['processed_review', 'sentiment_label'], inplace=True)
    df_test.dropna(subset=['processed_review', 'sentiment_label'], inplace=True)
    
    df_cv['sentiment_label'] = df_cv['sentiment_label'].astype(int)
    df_test['sentiment_label'] = df_test['sentiment_label'].astype(int)

    X_cv = df_cv['processed_review'].to_numpy()
    y_cv = df_cv['sentiment_label'].to_numpy()
    
    X_final_test = df_test['processed_review'].to_numpy()
    y_final_test = df_test['sentiment_label'].to_numpy()

    print(f"\n[데이터 준비 완료]")
    print(f" - 교차 검증용 데이터 (Train+Val): {len(X_cv)}개")
    print(f" - 최종 평가용 데이터 (Final Test): {len(X_final_test)}개")

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
            
    print("CV 데이터 토큰화 중")
    cv_encodings = tokenizer(X_cv.tolist(), truncation=True, padding=True, max_length=128)
    cv_dataset = ReviewDataset(cv_encodings, y_cv)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    print(f"\n{'='*20} {N_FOLDS}-Fold Cross Validation Start {'='*20}")

    scaler = torch.amp.GradScaler('cuda')

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_cv, y_cv)):
        print(f"\n--- [FOLD {fold+1}/{N_FOLDS}] ---")
        
        train_subsampler = Subset(cv_dataset, train_idx)
        val_subsampler = Subset(cv_dataset, val_idx)
        
        num_workers = 4
        train_loader = DataLoader(train_subsampler, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_subsampler, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
        
        print(f"Fold 학습 데이터: {len(train_subsampler)} / 검증 데이터: {len(val_subsampler)}")

        model = ElectraForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=4, use_safetensors=True)
        model.to(DEVICE)
        
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        total_steps = len(train_loader) * EPOCHS_PER_FOLD
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        best_fold_acc = 0.0
        
        for epoch in range(EPOCHS_PER_FOLD):
            model.train()
            train_loss = 0
            print(f"Fold {fold+1} Epoch {epoch+1} 학습 시작")
            
            for batch in tqdm(train_loader, desc="Training", leave=False):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
                attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
                labels = batch['labels'].to(DEVICE, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            val_loss, val_acc = evaluate_model(model, val_loader, DEVICE)
            print(f" -> [완료] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}\n")
            
            if val_acc > best_fold_acc:
                best_fold_acc = val_acc
                save_path = os.path.join(save_dir_base, f'best_sentiment_model_fold{fold+1}')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)

        print(f"Fold {fold+1} Best Accuracy: {best_fold_acc:.4f}")
        fold_results.append(best_fold_acc)
        
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    print(f"\n{'='*20} Cross Validation Result {'='*20}")
    print(f"각 Fold 별 정확도: {fold_results}")
    print(f"평균 정확도: {np.mean(fold_results):.4f}")
