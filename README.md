# LLM 기반 필터링을 이용한 한국어 쇼핑 리뷰 감성 분석

[![Paper](https://img.shields.io/badge/Paper-JICS_2026-blue.svg)](https://doi.org/10.7472/jksii.2026.27.3.35)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

<p align="center">
  <img src="./Overall framework.png" alt="framework" width="600">
</p>


네이버 쇼핑 리뷰의 평점과 텍스트가 일치하는지를 Gemini로 검토해 학습 데이터를 정제하고, 이를 바탕으로 4단계 감성 분류 모델을 학습하는 프로젝트입니다. 단순히 모델 구조를 바꾸기보다 데이터의 라벨 노이즈를 줄이는 데이터 중심 접근법에 초점을 둡니다.

본 저장소는 다음 논문의 구현을 포함합니다.

> **LLM 기반 필터링을 이용한 리뷰 데이터 중심 다중 클래스 감성 분석**
> Hyun-il Kim, Seon-jeong Hwang, and Choon-sung Nam, *Journal of Internet Computing and Services*, 2026.

## 주요 기능

- 네이버 쇼핑 리뷰의 기본 텍스트 정제와 중복 리뷰 제거
- 1·2·4·5점 리뷰를 4개 감성 레이블로 변환
- 점수별 상세 가이드라인을 이용한 Gemini 기반 평점-텍스트 정합성 필터링
- 정제 데이터의 80:20 계층 분할
- KcELECTRA 기반 5-Fold 교차 검증 학습
- Soft Voting 앙상블을 이용한 hold-out 데이터 평가
- Accuracy, Precision, Recall, F1-score 및 혼동 행렬 출력

## 처리 흐름

```text
원본 리뷰
  └─ 텍스트 정제 · 중복 제거 · 3점 제외
       └─ 점수별 Gemini 정합성 필터링
            └─ 정제 데이터셋 생성
                 ├─ 80%: 5-Fold 학습
                 └─ 20%: 최종 hold-out 평가
```

필터링은 [filtering_guideline_ko.txt](src/filtering_guideline_ko.txt)의 점수별 기준을 그대로 사용합니다. 예를 들어 1점은 제품 사용 불가나 강한 구매 만류가 있어야 하고, 4점은 전반적 만족과 함께 비핵심 요소에 대한 경미한 아쉬움이 있어야 통과합니다.


## 실험 결과

논문에서는 원본 20만 건의 리뷰 중 149,884건을 정제 데이터셋으로 구성했습니다. 최종 데이터 분포는 1점 20,014건, 2점 51,412건, 4점 11,656건, 5점 66,802건입니다.

| 모델 | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| BERT | 0.82 | 0.69 | 0.81 |
| KcBERT | 0.82 | 0.71 | 0.82 |
| KcELECTRA | **0.84** | **0.73** | **0.83** |

KcELECTRA는 정제 데이터에서 가장 높은 Weighted F1-score 0.83을 기록했습니다. 논문의 원본 데이터 BERT 기준 Macro F1-score 0.53과 비교하면, 데이터 정제 후 BERT의 Macro F1-score는 0.69로 향상되었습니다.

## 저장소 구조

```text
geminiAPI/
├── src/
│   ├── prepare_dataset.py          # 정제, Gemini 필터링, 데이터 분할
│   ├── filtering_guideline_ko.txt  # 점수별 필터링 기준
│   ├── train.py                    # KcELECTRA 5-Fold 학습
│   └── eval.py                     # 앙상블 평가
├── data/
│   ├── raw/                        # 원본 리뷰 데이터
│   ├── interim/                    # 정제·필터링 중간 결과
│   └── final/                      # 학습·평가용 최종 CSV
├── artifacts/                      # 모델 가중치와 평가 결과
├── requirements.txt
└── README.md
```

`data/`, `artifacts/`, `.env`는 Git에 포함하지 않습니다.

## 설치

```bash
git clone https://github.com/akadjsam/Korean-Text-Sentiment-Analysis.git
cd Korean-Text-Sentiment-Analysis

python -m venv .venv
```

가상환경을 활성화합니다.

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

의존성을 설치합니다.

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

주요 의존성은 PyTorch, Transformers, pandas, scikit-learn, Google Generative AI, Matplotlib, Seaborn입니다.

## 데이터 및 API 키 준비

원본 데이터는 [bab2min/corpus의 Naver Shopping Review Dataset](https://github.com/bab2min/corpus/tree/master/sentiment)을 내려받아 다음 경로에 둡니다.

```text
data/raw/naver_shopping.txt
```

Gemini 필터링을 실행하려면 프로젝트 루트에 `.env` 파일을 만들고 API 키를 설정합니다.

```text
API_KEY=your_gemini_api_key
```

## 실행

### 1. 데이터셋 생성

아래 명령은 텍스트 정제, 점수별 Gemini 필터링, 80:20 분할을 순서대로 수행합니다.

```bash
python src/prepare_dataset.py --filter
```

결과 파일은 다음 경로에 생성됩니다.

```text
data/final/train_for_kfold.csv
data/final/test_final_holdout.csv
```

`--filter`는 Gemini API를 호출합니다. API 비용과 실행 시간을 확인하려면 먼저 이 옵션 없이 실행해 정제 단계만 검토할 수 있습니다.

```bash
python src/prepare_dataset.py
```

### 2. 모델 학습

```bash
python src/train.py
```

학습된 각 Fold의 모델은 `artifacts/saved_models/` 아래에 저장됩니다. 기본 설정은 KcELECTRA, 최대 토큰 길이 128, 배치 크기 32, 학습률 5e-5, 5 Epoch입니다.

### 3. 앙상블 평가

```bash
python src/eval.py
```

평가 결과는 `artifacts/ensemble_predictions_result.csv`에 저장되며, hold-out 데이터의 분류 보고서와 혼동 행렬을 출력합니다.

## Limitations

- 인접 점수(1점/2점, 4점/5점)는 표현이 유사해 구분이 어렵습니다.
- 1점과 4점 클래스의 데이터 수가 상대적으로 적어 클래스 불균형이 남아 있습니다.
- LLM 필터링 결과는 모델 버전, 프롬프트, API 응답 환경에 따라 달라질 수 있습니다.
- 전체 데이터 필터링에는 Gemini API 호출 비용과 시간이 필요합니다.

## Citation

이 연구 또는 코드를 활용한 경우 다음 논문을 인용해 주세요.

```bibtex
@article{kim2026llm_filtering,
  author  = {Kim, Hyun-il and Hwang, Seon-jeong and Nam, Choon-sung},
  title   = {A Data-centric Approach for Multi-class Sentiment Analysis in Reviews using LLM-based Filtering},
  journal = {Journal of Internet Computing and Services},
  volume  = {27},
  number  = {3},
  pages   = {35--44},
  year    = {2026},
  doi     = {10.7472/jksii.2026.27.3.35}
}
```

## 참고 자료

- [Naver Shopping Review Dataset](https://github.com/bab2min/corpus/tree/master/sentiment)
- [KcELECTRA](https://github.com/Beomi/KcELECTRA)
- [Gemini API](https://ai.google.dev/)
