#!/usr/bin/env python3
"""
1단계: 한국어 감정 단어를 영어로 번역하여 en 컬럼 추가
emotions.tsv 파일을 읽어서 각 한국어 단어를 영어로 번역하고,
en 컬럼을 추가한 새로운 TSV 파일을 생성합니다.
"""

import csv
import sys
from pathlib import Path
from deep_translator import GoogleTranslator

# 스크립트가 있는 디렉토리 기준으로 경로 설정
SCRIPT_DIR = Path(__file__).parent
INPUT_FILE = SCRIPT_DIR / "emotions.tsv"
OUTPUT_FILE = SCRIPT_DIR / "emotions_with_translation.tsv"


def translate_korean_to_english(text: str) -> str:
    """
    한국어 텍스트를 영어로 번역합니다.

    Args:
        text: 번역할 한국어 텍스트

    Returns:
        번역된 영어 텍스트 (실패 시 원본 반환)
    """
    try:
        translator = GoogleTranslator(source="ko", target="en")
        translated = translator.translate(text)
        return translated.strip()
    except Exception as e:
        print(f"번역 실패: '{text}' - {e}", file=sys.stderr)
        return text


def main():
    """메인 함수"""
    print(f"입력 파일: {INPUT_FILE}")
    print(f"출력 파일: {OUTPUT_FILE}")
    print("번역을 시작합니다...")

    rows = []
    failed_translations = []

    # TSV 파일 읽기
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ko_word = row["ko"].strip()
            key = row["key"].strip()

            # 헤더 행이거나 빈 행은 건너뛰기
            if not ko_word or ko_word == "ko":
                continue

            # 번역 수행
            en_word = translate_korean_to_english(ko_word)

            # 번역 결과가 원본과 같으면 실패로 간주 (한국어가 그대로 남아있는 경우)
            if en_word == ko_word and ko_word:
                failed_translations.append(ko_word)

            rows.append({"ko": ko_word, "key": key, "en": en_word})

            print(f"  {ko_word} -> {en_word}")

    # 결과 저장
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["ko", "key", "en"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n번역 완료! 결과가 {OUTPUT_FILE}에 저장되었습니다.")
    print(f"총 {len(rows)}개 단어 번역됨")

    if failed_translations:
        print(
            f"\n경고: 다음 {len(failed_translations)}개 단어의 번역이 실패했거나 확인이 필요합니다:"
        )
        for word in failed_translations:
            print(f"  - {word}")
    else:
        print("\n모든 번역이 성공적으로 완료되었습니다!")


if __name__ == "__main__":
    main()
