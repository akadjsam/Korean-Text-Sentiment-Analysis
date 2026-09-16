"""논문 실험용 데이터셋 생성 스크립트."""

import argparse
import concurrent.futures
import json
import os
import re
from pathlib import Path

import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_TO_RATING = {0: 1, 1: 2, 2: 4, 3: 5}
GUIDELINE_PATH = Path(__file__).with_name("filtering_guideline_ko.txt")


def clean_review(text: object) -> str:
    text = str(text)
    text = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s.,!?]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_and_clean(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep="\t", header=None, names=["rating", "review"])
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["processed_review"] = df["review"].map(clean_review)
    df = df[df["processed_review"].str.len() > 0].drop_duplicates("processed_review")
    df = df[df["rating"].isin(LABEL_TO_RATING.values())].copy()
    rating_to_label = {rating: label for label, rating in LABEL_TO_RATING.items()}
    df["sentiment_label"] = df["rating"].map(rating_to_label).astype(int)
    return df[["rating", "review", "processed_review", "sentiment_label"]]


def load_filtering_guidelines(path: Path) -> dict[int, str]:
    """점수별 필터링 가이드라인을 읽는다."""
    content = path.read_text(encoding="utf-8-sig")
    matches = re.finditer(
        r"(?ms)^(?P<rating>[1245])점\s*$\n(?P<guide>.*?)(?=^[1245]점\s*$|\Z)",
        content,
    )
    guidelines = {
        int(match["rating"]): f"{match['rating']}점\n{match['guide'].strip()}"
        for match in matches
    }
    missing = set(LABEL_TO_RATING.values()) - set(guidelines)
    if missing:
        raise ValueError(f"Missing score guidelines in {path}: {sorted(missing)}")
    return guidelines


def make_prompt(guide: str, reviews: list[str]) -> str:
    serialized = json.dumps(reviews, ensure_ascii=False)
    return f"""{guide}

[입력 리뷰]
{serialized}

출력:"""


def classify_batch(model: genai.GenerativeModel, guide: str, reviews: list[str]) -> list[bool]:
    try:
        response = model.generate_content(make_prompt(guide, reviews))
        match = re.search(r"\[[\s\S]*?\]", response.text)
        values = json.loads(match.group(0)) if match else []
        if len(values) == len(reviews) and all(isinstance(value, str) for value in values):
            return [value.strip() == "일치" for value in values]
    except Exception as error:
        print(f"Gemini batch failed: {error}")
    return [False] * len(reviews)


def filter_with_gemini(
    df: pd.DataFrame, api_key: str, guidelines: dict[int, str], batch_size: int, workers: int
) -> pd.DataFrame:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    accepted_frames = []

    for label, rating in LABEL_TO_RATING.items():
        class_df = df[df["sentiment_label"] == label].reset_index(drop=True)
        batches = [class_df.iloc[index:index + batch_size] for index in range(0, len(class_df), batch_size)]
        decisions: list[bool] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(classify_batch, model, guidelines[rating], batch["processed_review"].tolist())
                for batch in batches
            ]
            for future in tqdm(futures, desc=f"Filtering rating {rating}"):
                decisions.extend(future.result())
        accepted_frames.append(class_df.loc[decisions])

    return pd.concat(accepted_frames, ignore_index=True)


def save_final_split(df: pd.DataFrame, data_dir: Path) -> None:
    final_dir = data_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["sentiment_label"]
    )
    train_df.to_csv(final_dir / "train_for_kfold.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(final_dir / "test_final_holdout.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the paper's filtered review dataset.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "raw" / "naver_shopping.txt")
    parser.add_argument("--filter", action="store_true", help="Run Gemini filtering and create the final split.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    cleaned = load_and_clean(args.input)
    interim_dir = PROJECT_ROOT / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(interim_dir / "cleaned_reviews.csv", index=False, encoding="utf-8-sig")
    print(f"Cleaned reviews: {len(cleaned):,}")

    if not args.filter:
        print("Cleaning complete. Re-run with --filter to call Gemini and create the final split.")
        return

    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("Set API_KEY in .env before running Gemini filtering.")
    guidelines = load_filtering_guidelines(GUIDELINE_PATH)
    filtered = filter_with_gemini(cleaned, api_key, guidelines, args.batch_size, args.workers)
    filtered.to_csv(interim_dir / "filtered_reviews.csv", index=False, encoding="utf-8-sig")
    save_final_split(filtered, PROJECT_ROOT / "data")
    print(f"Accepted reviews: {len(filtered):,}")


if __name__ == "__main__":
    main()
