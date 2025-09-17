from google import genai
#from google.genai import types
import google.generativeai as genai
import os
import json
from typing import List
import re
import pandas as pd
from tqdm import tqdm
import time # API 요청 간 딜레이를 위함
import concurrent.futures

class LlmAugmentation():
    def __init__(
            self,
            temperature: float = 1.0,
            candidate_count: int = 1
    ):

        genai.configure(api_key="Input your key") # API 키 입력
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            #system_instruction="당신은 뛰어난 언어 능력을 가진 문장가입니다."
            system_instruction = "당신은 네이버 쇼핑 리뷰 데이터를 증강하는 최고의 전문가입니다."
        )
        # 2. 생성 관련 설정을 GenerationConfig 객체로 만듭니다.
        self.generation_config = genai.GenerationConfig(
            temperature=temperature,
            candidate_count=candidate_count
        )
        self.safety_settings = {
            'HATE': 'BLOCK_NONE',
            'HARASSMENT': 'BLOCK_NONE',
            'SEXUAL': 'BLOCK_NONE',
            'DANGEROUS': 'BLOCK_NONE'
        }

    def generate_paraphrased_sentence(self, sentences: List[str]) -> List[str]:
        """문장 배열을 받아, 각 문장별로 의미는 동일하되 표현은 다른 문장으로 변형합니다."""

        # 입력 문장들을 JSON 배열 형식의 문자열로 변환
        input_sentences_str = json.dumps(sentences, ensure_ascii=False, indent=4)

        prompt = f"""
            당신은 네이버 쇼핑 리뷰 데이터를 증강하는 역할을 합니다.
            아래의 리뷰들을 같은 별점 톤을 유지하면서 표현만 다양하게 바꿔주세요.

            조건:
            - 반드시 입력 문장과 동일한 별점을 유지해야 합니다.
            - 숫자, 브랜드명, 고유명사(지명, 기관명, 인명 등)는 절대 바꾸지 마세요.
            - 결과물은 원래 문장과 내용 및 구조가 달라야 합니다. 다양한 상황을 가정하여 새로운 맥락을 부여하세요.
            - 결과는 다른 설명 없이, 오직 JSON 형식의 문자열 리스트로만 반환하세요.
            - 출력 문장에는 (4점)과 같은 레이블을 표시하지 마세요.
            별점별 톤 가이드:
            - 1점(강한 불만): 제품의 치명적인 결함, 배송 문제, 잘못된 설명 등 구체적인 최악의 경험을 추가하여 문장을 만드세요.
            - 2점(약한 불만): 기대치에 미치지 못하는 성능, 사소한 불편함, 아쉬운 마감 처리 등을 언급하며 문장을 만드세요.
            - 4점(대체로 만족): 핵심 기능에 대한 칭찬과 함께, '향, 포장, 배송, 특정 부가기능' 등 사소한 아쉬움을 하나 섞어서 문장을 만드세요.
            - 5점(매우 만족): 제품을 사용하며 얻은 긍정적인 결과나 삶의 변화, 재구매 의사, 주변인에게 추천하는 내용 등을 추가하여 문장을 만드세요.

            --- 예시 ---
            입력:
            [
                "가볍고 식단관리 하기에 딱 조절하기 좋습니다(5점)",
                "폭식하구 깔끔하니 고급지네요(5점)",
                "재질도 부드럽고 디자인도 예뻐요(4점)",
                "회사에서 쓸 목적으로 샀는데 휴대성 좋고 향도 좋아요 찢기가 조금 힘들지만 다른 제품도 다 그렇더라구요 괜찮요(4점)",
                "사용기기 상당히 조잡하고 싼티가남 불쾌한 접착제냄새가 심함 초점이 잘 맞지않아 어지러움 화면 크기 호환성은 좋으나 이어폰 꽂기에 불편하고 고정할때 별다른 완충장치가 없음(2점)",
                "후기에서 귀안은 확인가능하지만 보면서는 못파요 무슨 뜻인지 알았음확인용으로 쓰세용 모공 피지도 봄(2점)",
                "자르기 불편해요 맞는지 의심스러워요(1점)",
                "지루성피부염에 좋다는 샴푸 많이 써봤는데 이제품은 만족도가 많이 떨어집니다 개운하게 감기지도 않고 가려움 더 심하네요 샴푸개 다 새서 왔어요 추가 마개 필요한듯 하네요(1점)"
            ]

            출력:
            [
                "이 제품 덕분에 2주 만에 3kg 감량했어요. 매일 아침 도시락 싸는 시간이 즐거워질 줄은 몰랐네요. 다이어트하는 친구들에게 강력 추천합니다.",
                "주방에 놓았을 뿐인데 인테리어가 확 사는 느낌이에요. 성능은 말할 것도 없고, 디자인이 너무 고급스러워서 볼 때마다 기분이 좋아집니다. 다음엔 집들이 선물로 구매하려고요.",
                "소재가 정말 부들부들해서 피부에 닿는 느낌이 좋아요. 디자인도 화면에서 본 그대로 예쁜데, 포장이 조금 허술하게 와서 그 점은 아쉽네요. 제품 자체는 만족합니다.",
                "사무실 책상에 두고 쓰기 딱 좋은 사이즈라 휴대성이 정말 뛰어나요. 은은한 향 덕분에 일할 때 기분 전환도 되고요. 다만, 생각보다 빨리 닳는 것 같아서 그 점이 조금 아쉽습니다.",
                "기대를 많이 했는데, 전체적인 마감이 아쉽네요. 플라스틱 재질이 너무 가벼워서 장난감 같고, 처음 개봉했을 때 났던 화학약품 냄새가 아직도 다 안 빠졌어요. 가격을 생각하면 어쩔 수 없나 싶기도 하네요.",
                "연결은 잘 되는데, 이걸 보면서 실제로 귀를 관리하기는 거의 불가능에 가깝네요. 화면 딜레이가 미세하게 있어서 답답하고, 차라리 그냥 확인용으로만 가끔 쓰는 게 나을 것 같아요. 생각했던 것만큼 활용도가 높진 않아요.",
                "이거 절단용으로 나온 제품 맞나요? 아무리 힘을 줘도 잘리지도 않고, 오히려 비싼 재료만 다 망가뜨렸습니다. 결국 원래 쓰던 가위로 겨우 잘라냈는데, 이건 도저히 쓸 물건이 못 되네요. 환불받고 싶습니다.",
                "지루성 두피 때문에 믿고 구매했는데, 쓰고 나서 머리가 더 가렵고 비듬이 폭발했습니다. 설상가상으로 배송 온 박스를 열어보니 샴푸가 반쯤 새어 나와 다른 물건까지 엉망이 되었어요. 제품 효과도 없고 배송도 최악이라 돈만 버렸네요."
            ]
            --- 예시 끝 ---

            이제 다음 문장들을 재창조해주세요.

            입력:
            {input_sentences_str}

            출력:
        """

        try:
            response = self.model.generate_content(
                prompt,
                safety_settings=self.safety_settings,
                generation_config=self.generation_config
            )
            response_text = response.text.strip()

            # 모델이 응답에 ```json ... ``` 같은 마크다운을 포함할 경우 대비
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group()
                paraphrased_sentences = json.loads(json_text)

                # 입력과 출력의 개수가 다를 경우 원본을 반환하여 데이터 유실 방지
                if len(paraphrased_sentences) == len(sentences):
                    return paraphrased_sentences
                else:
                    print("Warning: 입출력 문장 개수가 다릅니다. 원본 배치를 반환합니다.")
                    return sentences
            else:
                print("Warning: 응답에서 JSON 배열을 찾을 수 없습니다. 원본 배치를 반환합니다.")
                return sentences

        except Exception as e:
            print(f"Error generating content: {e}")
            # print("Response text:", response_text if 'response_text' in locals() else "N/A")
            return sentences  # 에러 발생 시 원본 문장 리스트 반환


#
# def apply_augmentation_to_reviews(df, text_column, batch_size, repeat=1):
#     """
#     특정 텍스트 컬럼을 repeat 배만큼 증강합니다.
#     """
#     llm_aug = LlmAugmentation()
#     df_aug = df.copy()
#
#     original_texts = df[text_column].tolist()
#     #augmented_texts = []
#
#     # for _ in range(repeat):  # 여러 번 증강
#     #     for i in tqdm(range(0, len(original_texts), batch_size), desc=f"리뷰 증강 진행 중 (repeat={repeat})"):
#     #         batch = original_texts[i:i + batch_size]
#     #         paraphrased_batch = llm_aug.generate_paraphrased_sentence(batch)
#     #         augmented_texts.extend(paraphrased_batch)
#     #
#     # df_aug[text_column] = augmented_texts
#     # return df_aug
#     # 'repeat' 횟수만큼 전체 데이터에 대한 증강을 반복 실행
#
#     all_augmented_texts = []
#     for r_idx in range(repeat):
#         current_run_texts = []
#         for i in tqdm(range(0, len(original_texts), batch_size), desc=f"리뷰 증강 진행 중 (반복 {r_idx + 1}/{repeat})"):
#             batch = original_texts[i:i + batch_size]
#             paraphrased_batch = llm_aug.generate_paraphrased_sentence(batch)
#             current_run_texts.extend(paraphrased_batch)
#         all_augmented_texts.extend(current_run_texts)
#
#     # 1. 원본 데이터프레임을 'repeat' 횟수만큼 복제하여 길이를 맞춥니다.
#     #    예: 100개 df를 repeat=2 하면 200개짜리 df가 됨
#     df_repeated = pd.concat([df] * repeat, ignore_index=True)
#
#     # 2. 복제된 데이터프레임의 텍스트 컬럼을 증강된 텍스트로 교체합니다.
#     #    이제 df_repeated(26200개)와 all_augmented_texts(26200개)의 길이가 일치합니다.
#     df_repeated[text_column] = all_augmented_texts
#
#     return df_repeated

def apply_augmentation_to_reviews(df, text_column, batch_size, repeat=1):
    llm_aug = LlmAugmentation()

    original_texts = df[text_column].tolist()
    all_augmented_texts = []

    for r_idx in range(repeat):
        # 1. 작업을 배치 단위로 나눔
        batches = [original_texts[i:i + batch_size] for i in range(0, len(original_texts), batch_size)]

        # 증강된 배치를 순서대로 저장할 리스트
        paraphrased_results = [None] * len(batches)

        # 2. ThreadPoolExecutor로 병렬 작업 실행
        # max_workers는 동시에 보낼 요청 수 (예: 10~50 사이에서 조절)
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:

            # 각 배치에 대한 작업을 executor에 제출(submit)
            future_to_index = {
                executor.submit(llm_aug.generate_paraphrased_sentence, batch): i
                for i, batch in enumerate(batches)
            }

            # tqdm으로 진행 상황 확인
            for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(batches),
                               desc=f"리뷰 증강 진행 중 (반복 {r_idx + 1}/{repeat})"):
                index = future_to_index[future]
                try:
                    # 작업 결과(증강된 문장 리스트)를 원래 순서에 맞게 저장
                    paraphrased_results[index] = future.result()
                except Exception as exc:
                    print(f'배치 {index} 생성 중 예외 발생: {exc}')
                    # 실패 시 원본 배치로 채움
                    paraphrased_results[index] = batches[index]

        # 3. 순서대로 저장된 결과들을 하나의 리스트로 합침
        current_run_texts = [sentence for batch in paraphrased_results for sentence in batch]
        all_augmented_texts.extend(current_run_texts)

    # 이후 로직은 동일
    df_repeated = pd.concat([df] * repeat, ignore_index=True)
    df_repeated[text_column] = all_augmented_texts

    return df_repeated

def save_augmented_dataset(df, text_column, filename):
    """중복 제거 후 증강된 데이터셋을 CSV로 저장"""
    # 원본과 증강된 리뷰 내용이 같은 행은 제거
    df.drop_duplicates(subset=[text_column], inplace=True)
    df.reset_index(drop=True).to_csv(filename, index=False, encoding='utf-8')
    print(f"\n중복 제거 후 총 {len(df)}개의 데이터를 '{filename}'에 저장했습니다.")


if __name__ == "__main__":
    original_train_df = pd.read_csv('../data file/train.csv')

    # 증강할 라벨 선택
    labels_to_augment = [0,2]
    # 해당 라벨을 가진 데이터만 필터링
    # df_to_augment = original_train_df[original_train_df['sentiment_label'].isin(labels_to_augment)].copy()
    # print(f"\n라벨 {labels_to_augment}에 해당하는 데이터 {len(df_to_augment)}개를 증강합니다.")
    # # 데이터 증강 실행 (리뷰 텍스트가 있는 컬럼 이름 지정)
    # # API 비용이 발생할 수 있으니 주의하세요. 테스트 시에는 일부 데이터만 사용하세요.
    # # 예: augmented_df = apply_augmentation_to_reviews(original_train_df.head(10), text_column='processed_review', batch_size=5)
    # augmented_df = apply_augmentation_to_reviews(
    #     df_to_augment,
    #     text_column='processed_review',
    #     batch_size=8  # 배치 크기는 API 정책에 맞춰 조절
    # )
    # # 기존 데이터와 증강된 데이터를 합침
    # combined_df = pd.concat([original_train_df, augmented_df], ignore_index=True)

    # 클래스 0은 2배 증강
    df0 = original_train_df[original_train_df['sentiment_label'] == 0].copy()
    aug0 = apply_augmentation_to_reviews(df0, text_column='processed_review', batch_size=32, repeat=1)

    # 클래스 2는 3배 증강
    df2 = original_train_df[original_train_df['sentiment_label'] == 2].copy()
    aug2 = apply_augmentation_to_reviews(df2, text_column='processed_review', batch_size=32, repeat=2)

    # 합치기
    combined_df = pd.concat([original_train_df, aug0, aug2], ignore_index=True)


    # 저장 파일 이름과 방식을 CSV로 변경
    save_augmented_dataset(
        combined_df,
        text_column='processed_review',
        filename='train_create_API.csv'  # 원문과 많이 다른 문장을 생성
    )