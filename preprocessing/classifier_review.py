import os
import pandas as pd
import google.generativeai as genai
from tqdm import tqdm
import json
import re
import concurrent.futures
from dotenv import load_dotenv

# .env 파일에서 API 키를 로드합니다.
load_dotenv()
KEY = os.environ.get("API_KEY")
genai.configure(api_key=KEY)

# --- 설정 (사용자 환경에 맞게 수정) ---
BATCH_SIZE = 16  # 한 번의 API 호출에 포함할 리뷰 수
MAX_WORKERS = 5  # 동시에 실행할 API 요청의 수

# 입출력 파일 및 컬럼 정보
LABEL = 3  # 리뷰 점수가 아닌 레이블로 설정
INPUT_CSV_FILE = f'extract_lable_{LABEL}.csv'  # 원본 1점 리뷰 CSV 파일
REVIEW_COLUMN = 'processed_review'  # 리뷰 내용이 있는 열(column)의 이름
COMPLIANT_CSV_FILE = f'compliant_reviews_{LABEL}.csv'  # 가이드라인 '일치' 리뷰 저장 파일
NON_COMPLIANT_CSV_FILE = f'non_compliant_reviews_{LABEL}.csv'  # 가이드라인 '불일치' 리뷰 저장 파일

# 모델 설정
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-lite",  # 최신 모델 사용 권장
    system_instruction="당신은 주어진 가이드라인에 따라 리뷰 텍스트를 '일치' 또는 '불일치'로 분류하는 전문가입니다."
)

# API 안전 설정
safety_settings = {
    'HATE': 'BLOCK_NONE', 'HARASSMENT': 'BLOCK_NONE',
    'SEXUAL': 'BLOCK_NONE', 'DANGEROUS': 'BLOCK_NONE'
}


def classify_reviews_batch(batch_reviews: list):
    """
    리뷰 리스트(배치)를 받아, 각 리뷰에 대한 분류 결과('일치' 또는 '불일치') 리스트를 반환합니다.
    """
    # API에 전달할 프롬프트 생성
    input_reviews_str = json.dumps(batch_reviews, ensure_ascii=False, indent=2)

    prompt = f"""
    아래 JSON 리스트에 포함된 여러 개의 리뷰를 각각 분석해주세요.
    각 리뷰가 아래의 '5점 리뷰 가이드라인'에 완벽하게 부합하는지 판단하고 '일치' 또는 '불일치'로 분류해주세요.

    [5점 리뷰 가이드라인 (매우 강한 긍정)]
    - 주도 감정: 열광, 찬양, 감동, 절대적 만족, 기대를 초과하는 만족감.
    - 문체 특징: 열정적이고 확신에 찬 어조, 감탄사와 강조 표현을 자주 사용, 재구매 의사가 명확함.
    - 필터링 기준 (이 기준을 모두 만족해야 '일치'로 판단):
        1. 리뷰 내용 전체가 온전한 만족감과 긍정적인 내용으로만 구성되어야 함.
        2. 단점, 아쉬움, 개선점에 대한 표현이 단 하나도 포함되어서는 안 됨.
    - 매우 중요한 제외 기준 ('불일치' 판단 규칙):
        - 리뷰에 '하지만', '다만', '아쉬운 점', '조금', '약간', '~만 빼면', '~만 아니면' 같은 예외적이거나 아쉬움을 나타내는 단어가 조금이라도 포함되면 무조건 '불일치'로 분류해야 함.

    [분류 결과]
    - "일치": 리뷰가 위의 가이드라인을 완벽하게 만족하고, '제외 기준'에 해당하는 단어가 전혀 없을 때.
    - "불일치": 단 하나의 단점이나 아쉬움이라도 언급되거나, '제외 기준'의 단어가 포함될 때.

    [중요 규칙]
    - 반드시 입력된 리뷰와 '동일한 순서'와 '동일한 개수'의 JSON 리스트로만 답변해야 합니다.
    - 답변은 ["일치", "불일치", "일치", ...] 형태여야 합니다.
    - 다른 설명이나 코멘트는 절대로 추가하지 마세요.
    입력 리뷰 리스트:
    {input_reviews_str}

    출력:
    """
    try:
        response = model.generate_content(prompt, safety_settings=safety_settings)
        # print(response)
        response_text = response.text.strip()

        # 응답에서 JSON 리스트 부분만 정확히 추출
        json_match = max(re.findall(r'\[.*?\]', response_text, re.DOTALL), key=len, default=None)

        if json_match:
            predicted_labels = json.loads(json_match)
            if isinstance(predicted_labels, list) and len(predicted_labels) == len(batch_reviews):
                return predicted_labels

    except Exception as e:
        print(f"배치 처리 중 에러 발생: {e}") # 디버깅 필요 시 주석 해제
        pass

    # 실패 시 None 리스트 반환
    return [None] * len(batch_reviews)


# --- 메인 실행 로직 ---
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

    reviews_to_process = df[REVIEW_COLUMN].astype(str).tolist()
    batches = [reviews_to_process[i:i + BATCH_SIZE] for i in range(0, len(reviews_to_process), BATCH_SIZE)]
    all_classifications = []

    print(f"총 {len(reviews_to_process)}개의 리뷰를 {len(batches)}개의 배치로 나누어 처리합니다.")

    # ThreadPoolExecutor를 사용하여 API 요청을 병렬로 처리
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(classify_reviews_batch, batches), total=len(batches), desc="리뷰 필터링 중"))

    for batch_result in results:
        all_classifications.extend(batch_result)

    # 결과 분류 및 저장
    compliant_rows = []
    non_compliant_rows = []
    error_count = 0

    for i, classification in enumerate(all_classifications):
        if classification == "일치":
            compliant_rows.append(df.iloc[i])
        elif classification == "불일치":
            non_compliant_rows.append(df.iloc[i])
        else:
            error_count += 1

    print("\n--- 필터링 결과 요약 ---")
    print(f"가이드라인 '일치' 리뷰: {len(compliant_rows)}개")
    print(f"가이드라인 '불일치' 리뷰: {len(non_compliant_rows)}개")
    print(f"분류 실패/에러: {error_count}개")

    # '일치' 리뷰를 CSV 파일로 저장
    if compliant_rows:
        compliant_df = pd.DataFrame(compliant_rows)
        compliant_df.to_csv(COMPLIANT_CSV_FILE, index=False, encoding='utf-8-sig')
        print(f"\n가이드라인 '일치' 리뷰를 '{COMPLIANT_CSV_FILE}' 파일로 저장했습니다.")

    # '불일치' 리뷰를 CSV 파일로 저장
    if non_compliant_rows:
        non_compliant_df = pd.DataFrame(non_compliant_rows)
        non_compliant_df.to_csv(NON_COMPLIANT_CSV_FILE, index=False, encoding='utf-8-sig')
        print(f"가이드라인 '불일치' 리뷰를 '{NON_COMPLIANT_CSV_FILE}' 파일로 저장했습니다.")