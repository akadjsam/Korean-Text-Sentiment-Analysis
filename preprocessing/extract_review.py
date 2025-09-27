# -*- coding: utf-8 -*-

import pandas as pd

# --- 설정 ---
INPUT_FILE = 'preprocessed_reviews.csv'
OUTPUT_FILE = 'extract_result.csv'
TARGET_LABEL = 2 # 추출하고 싶은 레이블을 설정


def filter_and_save_reviews(input_path, output_path, label):
    """
    CSV 파일을 읽어 특정 레이블의 데이터만 필터링하여 새 CSV 파일로 저장합니다.
    """
    try:
        # 1. CSV 파일 읽기
        print(f"'{input_path}' 파일을 읽는 중입니다...")
        df = pd.read_csv(input_path)
        print(f"총 {len(df)}개의 데이터를 불러왔습니다.")

        # 2. 'sentiment_label'이 2인 데이터만 필터링
        print(f"'sentiment_label'이 {label}인 데이터를 필터링합니다...")
        filtered_df = df[df['sentiment_label'] == label]

        if filtered_df.empty:
            print(f"경고: 'sentiment_label'이 {label}인 데이터가 없습니다.")
            return

        print(f"필터링 결과: {len(filtered_df)}개의 데이터를 찾았습니다.")

        # 3. 필터링된 데이터를 새 CSV 파일로 저장
        filtered_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"필터링된 데이터를 '{output_path}' 파일로 성공적으로 저장했습니다.")

    except FileNotFoundError:
        print(f"오류: '{input_path}' 파일을 찾을 수 없습니다. 파일이 현재 폴더에 있는지 확인하세요.")
    except KeyError:
        print(f"오류: 파일에 'sentiment_label' 컬럼이 없습니다. 입력 파일의 컬럼명을 확인하세요.")
    except Exception as e:
        print(f"알 수 없는 오류가 발생했습니다: {e}")


# --- 메인 코드 실행 ---
if __name__ == "__main__":
    filter_and_save_reviews(INPUT_FILE, OUTPUT_FILE, TARGET_LABEL)