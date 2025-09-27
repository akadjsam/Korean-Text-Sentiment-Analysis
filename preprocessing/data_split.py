import pandas as pd
from sklearn.model_selection import train_test_split

# --- 데이터 로드 ---
try:
    df = pd.read_csv("naver_shopping.csv")
except FileNotFoundError:
    print("오류: naver_shopping.csv 파일을 찾을 수 없습니다.")
    exit() # 파일이 없으면 프로그램 종료

# --- 1. 단순 랜덤 분할 ---
print("scikit-learn을 사용한 단순 랜덤 분할을 시작합니다...")

# 특성(X)과 라벨(y) 분리
X = df.drop('sentiment_label', axis=1)
y = df['sentiment_label']

# 1단계: train(70%) / temp(30%) 분할
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 2단계: temp(30%) -> validation(15%) / test(15%) 분할
# temp의 50%는 전체의 15%에 해당
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# 분할 결과 확인
print("\n분할 완료!")
print(f"  - Train 세트: {X_train.shape[0]}개")
print(f"  - Validation 세트: {X_val.shape[0]}개")
print(f"  - Test 세트: {X_test.shape[0]}개")


# (선택) 분할된 데이터를 다시 합쳐서 CSV로 저장하려면
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv('train_0918.csv', index=False, encoding='utf-8-sig')
val_df.to_csv('validation_0918.csv', index=False, encoding='utf-8-sig')
test_df.to_csv('test_0918.csv', index=False, encoding='utf-8-sig')
print("\nCSV 파일 저장 완료!")

print("-" * 40)

# --- 2. (참고) 클래스 비율을 맞추는 분할 (Stratified) ---
print("\nscikit-learn의 'stratify' 옵션으로 클래스 비율을 맞춰 분할합니다...")

# stratify=y 옵션만 추가하면 됩니다.
X_train_s, X_temp_s, y_train_s, y_temp_s = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val_s, X_test_s, y_val_s, y_test_s = train_test_split(
    X_temp_s, y_temp_s, test_size=0.5, random_state=42, stratify=y_temp_s
)

# 분할 결과 확인
print("\nStratified 분할 완료!")
print(f"  - Train 세트: {X_train_s.shape[0]}개, 라벨 분포:\n{y_train_s.value_counts().sort_index().to_string()}")
print(f"  - Validation 세트: {X_val_s.shape[0]}개, 라벨 분포:\n{y_val_s.value_counts().sort_index().to_string()}")
print(f"  - Test 세트: {X_test_s.shape[0]}개, 라벨 분포:\n{y_test_s.value_counts().sort_index().to_string()}")