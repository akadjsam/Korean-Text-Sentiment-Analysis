import pandas as pd
import google.generativeai as genai
from tqdm import tqdm
# confusion_matrix를 추가로 import합니다.
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import asyncio
from tqdm.asyncio import tqdm_asyncio # 비동기 tqdm
import json
import re
import concurrent.futures

# 1. API 키 설정 ('' 안에 실제 API 키를 입력하세요)
genai.configure(api_key="Input your key")

# 2. 제로샷 추론을 위한 모델 설정
model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-lite",
            #system_instruction="당신은 뛰어난 언어 능력을 가진 문장가입니다."
            system_instruction = "당신은 네이버 쇼핑 리뷰 데이터를 4가지 평점으로 분류하는 전문가입니다."
        )

# 3. 라벨 맵 정의 (데이터셋의 숫자 라벨과 텍스트 라벨을 매핑)
label_map = {
    0: '매우 부정적',
    1: '부정적',
    2: '긍정적',
    3: '매우 긍정적'
}
safety_settings = {
            'HATE': 'BLOCK_NONE',
            'HARASSMENT': 'BLOCK_NONE',
            'SEXUAL': 'BLOCK_NONE',
            'DANGEROUS': 'BLOCK_NONE'
        }
reverse_label_map = {v: k for k, v in label_map.items()}
label_names = list(label_map.values())  # Confusion Matrix 시각화에 사용

# 4. 데이터 로드
try:
    test_df = pd.read_csv('../data file/test.csv')
    test_df.dropna(subset=['processed_review', 'sentiment_label'], inplace=True)
    test_df['sentiment_label'] = test_df['sentiment_label'].astype(int)
except FileNotFoundError:
    print("파일을 찾을 수 없습니다. 경로를 확인하세요.")
    exit()

# semaphore = asyncio.Semaphore(15)
# 5. 개별 API 요청을 처리하는 비동기 함수 정의
def get_predictions_batch(batch_reviews: list):
    """리뷰 리스트(배치)를 받아, 각 리뷰에 대한 감성 분류 결과 리스트를 반환"""

    # 입력 리뷰들을 JSON 배열 형식으로 프롬프트에 삽입
    input_reviews_str = json.dumps(batch_reviews, ensure_ascii=False, indent=2)

    prompt = f"""
    당신은 여러 개의 네이버 쇼핑 리뷰를 한 번에 분석하여 각각의 핵심 감정을 분류하는 전문가입니다.
    아래에 JSON 리스트 형식으로 주어진 각 리뷰를 읽고, ['매우 긍정적', '긍정적', '부정적', '매우 부정적'] 중 하나로 분류해주세요.

    [중요 규칙]
    - 반드시 입력된 리뷰와 '동일한 순서'와 '동일한 개수'로 된 JSON 리스트로만 답변해야 합니다.
    - 다른 설명이나 코멘트는 절대로 추가하지 마세요.

    이제 다음 리뷰 리스트를 분류해주세요.

    입력:
    {input_reviews_str}

    출력:
    """

    try:
        #response = model.generate_content(prompt)
        response = model.generate_content(
            prompt,
            safety_settings=safety_settings
        )
        response_text = response.text.strip()
        # 응답에서 JSON 리스트 부분만 추출
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group()
            predicted_labels_text = json.loads(json_text)

            # 입력과 출력의 개수가 다르면 에러로 간주
            if len(predicted_labels_text) == len(batch_reviews):
                return predicted_labels_text

    except Exception as e:
        # print(f"API 또는 파싱 에러: {e}") # 디버깅 시
        pass  # 에러 발생 시 아래에서 처리됨

    # 실패 시 입력 배치와 동일한 길이의 None 리스트 반환
    return [None] * len(batch_reviews)


# 2. 메인 로직
if __name__ == "__main__":
    test_df = pd.read_csv('../data file/test.csv').dropna(subset=['processed_review', 'sentiment_label'])

    batch_size = 32  # 한 번의 API 호출에 포함될 리뷰 수

    original_reviews = test_df['processed_review'].tolist()
    true_labels = test_df['sentiment_label'].tolist()

    # 작업을 배치 단위로 나눔
    batches = [original_reviews[i:i + batch_size] for i in range(0, len(original_reviews), batch_size)]

    all_predicted_labels = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # tqdm을 사용하여 진행률 표시
        results = list(tqdm(executor.map(get_predictions_batch, batches), total=len(batches), desc="배치 제로샷 추론 중"))

    # 결과 취합
    for batch_result in results:
        all_predicted_labels.extend(batch_result)

    # 결과 처리 및 평가
    final_predictions = []
    final_true_labels = []  # 공한 예측에 해당하는 실제 정답만 담을 리스트
    error_count = 0

    # all_predicted_labels는 API 응답 텍스트 리스트, true_labels는 전체 정답 숫자 리스트
    for i, pred_text in enumerate(all_predicted_labels):
        if pred_text and pred_text in reverse_label_map:
            # 파싱에 성공하면, 예측값과 실제 정답을 각각의 최종 리스트에 추가
            final_predictions.append(reverse_label_map[pred_text])
            final_true_labels.append(true_labels[i])
        else:
            # 실패하면 에러 카운트만 증가
            error_count += 1

    print(f"\n추론 완료! (성공: {len(final_predictions)} / 에러 또는 파싱 실패: {error_count})")

    # 8. 성능 평가
    if final_predictions:
        # 평가 함수에 길이가 동일하게 필터링된 final_true_labels와 final_predictions를 전달합니다.
        accuracy = accuracy_score(final_true_labels, final_predictions)

        print("\n\n--- Gemini Zero-shot 성능 평가 결과 ---\n")
        print(f"Accuracy (정확도): {accuracy:.4f}")

        print("\nClassification Report\n")
        report = classification_report(
            final_true_labels,  # <-- 수정된 부분
            final_predictions,
            target_names=label_names,
            digits=4
        )
        print(report)

        print("\nConfusion Matrix\n")
        cm = confusion_matrix(
            final_true_labels,  # <-- 수정된 부분
            final_predictions
        )
        print(cm)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix_async.png')
        print("\n혼동 행렬을 'confusion_matrix_async.png' 파일로 저장했습니다.")
    else:
        print("평가를 위한 유효한 예측 결과가 없습니다.")