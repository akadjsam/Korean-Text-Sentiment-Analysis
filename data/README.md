# 데이터 안내

```text
data/
├── raw/
│   └── naver_shopping.txt       # 원본 네이버 쇼핑 리뷰
├── interim/
│   ├── cleaned_reviews.csv      # 기본 정제 결과
│   └── filtered_reviews.csv     # Gemini 필터링 결과
└── final/
    ├── train_for_kfold.csv      # 80% 학습·교차 검증 데이터
    └── test_final_holdout.csv   # 20% 최종 평가 데이터
```

## 원본 데이터 준비

원본 리뷰는 [bab2min/corpus의 Naver Shopping Review Dataset](https://github.com/bab2min/corpus/tree/master/sentiment)에서 내려받아 다음 경로에 둡니다.

```text
data/raw/naver_shopping.txt
```

## 데이터셋 생성

프로젝트 루트에서 다음 명령을 실행합니다.

```bash
python src/prepare_dataset.py --filter
```

이 명령은 텍스트 정제, 점수별 Gemini 정합성 필터링, 80:20 계층 분할을 수행합니다. 필터링 기준은 `src/filtering_guideline_ko.txt`에 있으며, Gemini API 키는 루트의 `.env` 파일에 `API_KEY`로 설정해야 합니다.