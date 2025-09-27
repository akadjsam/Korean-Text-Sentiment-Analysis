# 전처리 결과 시각화
import pandas as pd
import re
from transformers import BertTokenizer
from collections import Counter

def clean_korean_text(text):
    """
    한글, 공백 외 문자 제거 및 연속 공백을 단일 공백으로 변환
    """
    cleaned = re.sub(r'[^가-힣\s]', '', str(text)) # 가~힣, 공백을 제외한 나머지 제거
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

def remove_stopwords(text, stopwords):
    """
    불용어 제거
    """
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)

def rating_to_sentiment(rating):
    """
    평점을 감성 라벨(0~3, 4가지)로 변환. 3점은 None으로 처리.
    """
    try:
        rating = int(rating)
        if rating == 5: return 3   # 매우 긍정적
        elif rating == 4: return 2 # 긍정적
        elif rating == 2: return 1 # 부정적
        elif rating == 1: return 0 # 매우 부정적
        else: return None
    except:
        return None

def load_stopwords(stopwords_file):
    """
    텍스트 파일에서 불용어 목록 로드
    """
    try:
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            stopwords = [line.strip() for line in f.readlines() if line.strip()]
        print(f"불용어 {len(stopwords)}개 로드 완료: {stopwords_file}")
        return stopwords
    except FileNotFoundError:
        print(f"불용어 파일을 찾을 수 없습니다: {stopwords_file}")
        print("기본 불용어를 사용합니다.")
        return ['은', '는', '이', '가', '고', '을', '를']
    except Exception as e:
        print(f"불용어 파일 로딩 오류: {e}")
        print("기본 불용어를 사용합니다.")
        return ['은', '는', '이', '가', '고', '을', '를']

def tokenize_with_bert(text, tokenizer, max_length=128):
    """
    BERT 토크나이저로 텍스트를 토큰화
    """
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    # 토큰 ID를 실제 토큰으로 변환
    input_ids = encoded['input_ids'].squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # 패딩 토큰이 아닌 실제 토큰만 필터링
    actual_tokens = []
    actual_token_ids = []
    for token_id, token in zip(input_ids, tokens):
        if token_id != tokenizer.pad_token_id:
            actual_tokens.append(token)
            actual_token_ids.append(token_id)

    return {
        'input_ids': input_ids,
        'attention_mask': encoded['attention_mask'].squeeze().tolist(),
        'tokens': tokens,  # 패딩 포함 전체 토큰
        'actual_tokens': actual_tokens,  # 패딩 제외 실제 토큰
        'actual_token_ids': actual_token_ids  # 패딩 제외 실제 토큰 ID
    }


def preprocess_naver_shopping_data(file_path, stopwords_file):
    """
    네이버 쇼핑 리뷰 데이터 전처리
    """

    # 데이터 로드
    try:
        df = pd.read_csv(file_path, sep='\t', header=None, names=['rating', 'review'])
        print(f"총 {len(df)}개의 리뷰 로드 완료")
    except Exception as e:
        print(f"파일 로딩 오류: {e}")
        return None, None

    # 텍스트 정제
    print("텍스트 정제")
    df['cleaned_review'] = df['review'].apply(clean_korean_text)
    df = df[df['cleaned_review'].str.len() > 0] # 빈 리뷰 제거

    # 중복 리뷰 제거
    print("중복 리뷰 제거")
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['cleaned_review'], keep='first')
    print(f"{before_dedup - len(df)}개 중복 리뷰 제거 완료, {len(df)}개 남음")

    # 불용어 제거
    print("불용어 제거")
    stopwords = load_stopwords(stopwords_file)
    df['processed_review'] = df['cleaned_review'].apply(lambda x: remove_stopwords(x, stopwords))

    # 감성 라벨 매핑
    print("감성 라벨 매핑")
    df['sentiment_label'] = df['rating'].apply(rating_to_sentiment)
    df = df.dropna(subset=['sentiment_label']) # 3점 리뷰 등 None 값 제거
    df['sentiment_label'] = df['sentiment_label'].astype(int)

    # 데이터 분포 확인
    sentiment_counts = df['sentiment_label'].value_counts().sort_index()
    sentiment_names = {0: '매우 부정적', 1: '부정적', 2: '긍정적', 3: '매우 긍정적'}
    print("\n감성 라벨 분포:")
    #print(sentiment_counts)
    for label, count in sentiment_counts.items():
        print(f"  {sentiment_names[label]} ({label}): {count}개")

    return df

def save_processed_data(df, output_path):
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"전처리된 데이터 저장 완료 : {output_path}")

if __name__ == "__main__":
    FILE_PATH = '../data file/naver_shopping.txt'
    STOPWORDS_FILE = 'stopwords-ko.txt'
    # 데이터 전처리
    processed_df = preprocess_naver_shopping_data(FILE_PATH, STOPWORDS_FILE)
    save_processed_data(processed_df, 'preprocessed_reviews.csv') # csv 형식으로 저장

    if processed_df is not None:
        # 전처리된 데이터프레임 확인
        print("\n전처리 완료된 데이터프레임 (상위 5개)")
        print(processed_df.head())

        # 샘플 토큰화 진행 및 결과 확인
        print("\n샘플 토큰화 결과")
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        for index, row in processed_df.head(3).iterrows():
            review_text = row['processed_review']
            tokens_result = tokenize_with_bert(review_text, tokenizer)

            print(f"\n{'='*50}")
            print(f"샘플 {index+1}")
            print(f"{'='*50}")
            print(f"원본 리뷰: {row['review']}")
            print(f"전처리된 리뷰: {review_text}")
            print(f"\n실제 토큰들 (패딩 제외): {tokens_result['actual_tokens']}")
            print(f"실제 토큰 개수: {len(tokens_result['actual_tokens'])}")
            print(f"\n토큰 ID (실제): {tokens_result['actual_token_ids']}")
            print(f"토큰 ID (패딩 포함, 일부): {tokens_result['input_ids'][:15]}...")
            print(f"전체 길이 (패딩 포함): {len(tokens_result['input_ids'])}")

            # 토큰별 ID
            print(f"\n토큰별 상세 분석:")
            for i, (token, token_id) in enumerate(zip(tokens_result['actual_tokens'], tokens_result['actual_token_ids'])):
                print(f"  {i+1:2d}: '{token}' (ID: {token_id})")
                if i >= 10:  # 처음 10개만 출력
                    print(f"  ... (총 {len(tokens_result['actual_tokens'])}개 토큰)")
                    break