
import os
import pandas as pd
import google.generativeai as genai
from tqdm import tqdm
import json
import re
import concurrent.futures
from dotenv import load_dotenv
load_dotenv()
KEY = os.environ.get("API_KEY")
genai.configure(api_key=KEY)
BATCH_SIZE = 8      # 한 번의 API 호출에 포함할 리뷰 수 (최대 100 권장)
MAX_WORKERS = 10     # 동시에 실행할 API 요청의 수 (네트워크 환경에 따라 조절)
# 입출력 파일 및 컬럼 정보

INPUT_CSV_FILE = 'extract_result.csv'  # 원본 4점 리뷰 CSV 파일 이름
REVIEW_COLUMN = 'processed_review'            # 리뷰 내용이 있는 열(column)의 이름
POSITIVE_CSV_FILE = 'positive_reviews(4).csv'  # 긍정 리뷰 저장 파일 이름
MIXED_CSV_FILE = 'mixed_reviews(4).csv'  # 부정 혼합 리뷰 저장 파일 이름

# 모델 설정
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-lite",
    system_instruction="당신은 리뷰 텍스트를 분석하여 '긍정' 또는 '부정 혼합'으로 분류하는 전문가입니다."
)

# API 안전 설정
safety_settings = {
    'HATE': 'BLOCK_NONE', 'HARASSMENT': 'BLOCK_NONE',
    'SEXUAL': 'BLOCK_NONE', 'DANGEROUS': 'BLOCK_NONE'
}


def classify_reviews_batch(batch_reviews: list):
    """
    리뷰 리스트(배치)를 받아, 각 리뷰에 대한 분류 결과('긍정' 또는 '부정 혼합') 리스트를 반환합니다.
    """
    # API에 전달할 프롬프트 생성
    # json.dumps를 사용하여 리뷰 리스트를 JSON 문자열로 변환
    input_reviews_str = json.dumps(batch_reviews, ensure_ascii=False, indent=2)

    prompt = f"""
    아래 JSON 리스트에 포함된 여러 개의 리뷰를 각각 분석해줘.
    각 리뷰가 '오직 긍정적인 내용'만 담고 있는지, 아니면 '부정적인 내용이 조금이라도 섞여 있는지' 판단해줘.

    [분류 기준]
    - "긍정": 리뷰 내용 전체가 긍정적일 때
    - "부정 혼합": 긍정적인 내용이 있더라도, 불만, 아쉬움, 단점 등 부정적인 뉘앙스가 조금이라도 포함될 때

    [중요 규칙]
    - 반드시 입력된 리뷰와 '동일한 순서'와 '동일한 개수'의 JSON 리스트로만 답변해야 합니다.
    - 답변은 ["긍정", "부정 혼합", "긍정", ...] 형태여야 합니다.
    - 다른 설명이나 코멘트는 절대로 추가하지 마세요.

    입력 리뷰 리스트:
    {input_reviews_str}

    출력:
    """
    try:
        response = model.generate_content(prompt, safety_settings=safety_settings)
        response_text = response.text.strip()

        # 응답에서 JSON 리스트 부분만 정확히 추출 (가장 긴 `[]` 블록을 찾음)
        json_match = max(re.findall(r'\[.*?\]', response_text, re.DOTALL), key=len, default=None)

        if json_match:
            predicted_labels = json.loads(json_match)
            # 입력과 출력의 개수가 일치하는지 확인
            if isinstance(predicted_labels, list) and len(predicted_labels) == len(batch_reviews):
                return predicted_labels

    except Exception as e:
        # API 호출 또는 JSON 파싱 중 에러 발생 시
        # print(f"배치 처리 중 에러 발생: {e}") # 디버깅 필요 시 주석 해제
        pass

    # 어떤 이유로든 실패하면, 배치 크기만큼 None 리스트를 반환하여 에러를 표시
    return [None] * len(batch_reviews)


# --- 2. 메인 실행 로직 ---
if __name__ == "__main__":
    try:
        df = pd.read_csv(INPUT_CSV_FILE)
        df.dropna(subset=[REVIEW_COLUMN], inplace=True)
    except FileNotFoundError:
        print(f"오류: 입력 파일 '{INPUT_CSV_FILE}'을 찾을 수 없습니다. 파일 경로와 이름을 확인해주세요.")
        exit()
    except KeyError:
        print(f"오류: CSV 파일에서 '{REVIEW_COLUMN}' 열을 찾을 수 없습니다. REVIEW_COLUMN 변수를 확인해주세요.")
        exit()

    # 리뷰 텍스트를 리스트로 변환
    reviews_to_process = df[REVIEW_COLUMN].astype(str).tolist()

    # 리뷰 리스트를 배치 크기 단위로 나눔
    batches = [reviews_to_process[i:i + BATCH_SIZE] for i in range(0, len(reviews_to_process), BATCH_SIZE)]

    all_classifications = []

    print(f"총 {len(reviews_to_process)}개의 리뷰를 {len(batches)}개의 배치로 나누어 처리합니다.")

    # ThreadPoolExecutor를 사용하여 API 요청을 병렬로 처리
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # tqdm을 사용하여 진행 상황 시각화
        results = list(tqdm(executor.map(classify_reviews_batch, batches), total=len(batches), desc="리뷰 분류 중"))

    # 병렬 처리된 결과들을 하나의 리스트로 합침
    for batch_result in results:
        all_classifications.extend(batch_result)

    # 결과 분류 및 저장
    positive_rows = []
    mixed_rows = []
    error_count = 0

    # 원본 데이터프레임(df)의 인덱스와 분류 결과를 함께 순회
    for i, classification in enumerate(all_classifications):
        if classification == "긍정":
            positive_rows.append(df.iloc[i])
        elif classification == "부정 혼합":
            mixed_rows.append(df.iloc[i])
        else:
            # API 응답이 없거나 파싱에 실패한 경우
            error_count += 1

    print("\n--- 분류 결과 요약 ---")
    print(f"긍정 리뷰: {len(positive_rows)}개")
    print(f"부정 혼합 리뷰: {len(mixed_rows)}개")
    print(f"분류 실패/에러: {error_count}개")

    # 긍정 리뷰를 CSV 파일로 저장
    if positive_rows:
        positive_df = pd.DataFrame(positive_rows)
        positive_df.to_csv(POSITIVE_CSV_FILE, index=False, encoding='utf-8-sig')
        print(f"\n긍정 리뷰를 '{POSITIVE_CSV_FILE}' 파일로 저장했습니다.")

    # 부정 혼합 리뷰를 CSV 파일로 저장
    if mixed_rows:
        mixed_df = pd.DataFrame(mixed_rows)
        mixed_df.to_csv(MIXED_CSV_FILE, index=False, encoding='utf-8-sig')
        print(f"부정 혼합 리뷰를 '{MIXED_CSV_FILE}' 파일로 저장했습니다.")