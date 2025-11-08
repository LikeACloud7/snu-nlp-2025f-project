#!/usr/bin/env python3
"""
2단계: 영어 단어를 NRC-VAD Lexicon에서 검색하여 VAD 값 매칭
emotions_with_translation.tsv 파일을 읽어서 각 영어 단어의
valence, arousal, dominance 값을 lexicon에서 찾아 추가합니다.
"""

import csv
from pathlib import Path

# 스크립트가 있는 디렉토리 기준으로 경로 설정
SCRIPT_DIR = Path(__file__).parent
INPUT_FILE = SCRIPT_DIR / "emotions_with_translation.tsv"
LEXICON_FILE = SCRIPT_DIR / "NRC-VAD-Lexicon-v2.1.txt"
OUTPUT_FILE = SCRIPT_DIR / "emotions.tsv"


def load_lexicon(lexicon_path: Path) -> dict:
    """
    NRC-VAD Lexicon 파일을 로드하여 딕셔너리로 반환합니다.

    Args:
        lexicon_path: lexicon 파일 경로

    Returns:
        {term: {'valence': float, 'arousal': float, 'dominance': float}} 형태의 딕셔너리
    """
    lexicon = {}

    print(f"Lexicon 파일 로딩 중: {lexicon_path}")

    with open(lexicon_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            term = row["term"].strip().lower()  # 대소문자 무시를 위해 소문자로 변환
            try:
                valence = float(row["valence"])
                arousal = float(row["arousal"])
                dominance = float(row["dominance"])

                # 같은 term이 여러 번 나올 수 있으므로, 첫 번째 것만 사용하거나 평균을 낼 수 있음
                # 여기서는 첫 번째 것만 사용
                if term not in lexicon:
                    lexicon[term] = {
                        "valence": valence,
                        "arousal": arousal,
                        "dominance": dominance,
                    }
            except (ValueError, KeyError):
                # 잘못된 행은 건너뛰기
                continue

    print(f"Lexicon 로딩 완료: {len(lexicon)}개 항목")
    return lexicon


def find_vad_values(en_word: str, lexicon: dict) -> tuple:
    """
    영어 단어에 대응하는 VAD 값을 lexicon에서 찾습니다.

    Args:
        en_word: 검색할 영어 단어
        lexicon: VAD lexicon 딕셔너리

    Returns:
        (valence, arousal, dominance) 튜플 또는 (None, None, None) (매칭 실패 시)
    """
    if not en_word or not en_word.strip():
        return (None, None, None)

    # 대소문자 무시하여 검색
    search_term = en_word.strip().lower()

    # 정확 일치 시도
    if search_term in lexicon:
        vad = lexicon[search_term]
        return (vad["valence"], vad["arousal"], vad["dominance"])

    # 부분 일치 시도 (예: "angry"가 "angry person"에 포함되는 경우)
    # 하지만 여기서는 정확 일치만 사용 (감정 단어는 보통 단일 단어이므로)

    return (None, None, None)


def main():
    """메인 함수"""
    print(f"입력 파일: {INPUT_FILE}")
    print(f"Lexicon 파일: {LEXICON_FILE}")
    print(f"출력 파일: {OUTPUT_FILE}")

    # Lexicon 로드
    lexicon = load_lexicon(LEXICON_FILE)

    # 입력 파일 읽기
    rows = []
    matched_count = 0
    failed_words = []

    print("\nVAD 값 매칭을 시작합니다...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ko_word = row["ko"].strip()
            key = row["key"].strip()
            en_word = row.get("en", "").strip()

            # 헤더 행이거나 빈 행은 건너뛰기
            if not ko_word or ko_word == "ko":
                continue

            # VAD 값 찾기
            valence, arousal, dominance = find_vad_values(en_word, lexicon)

            if valence is not None:
                matched_count += 1
                print(
                    f"  ✓ {en_word}: V={valence:.3f}, A={arousal:.3f}, D={dominance:.3f}"
                )
            else:
                failed_words.append((ko_word, en_word))
                print(f"  ✗ {en_word} (한국어: {ko_word}) - 매칭 실패")

            rows.append(
                {
                    "ko": ko_word,
                    "key": key,
                    "en": en_word,
                    "valence": f"{valence:.6f}" if valence is not None else "",
                    "arousal": f"{arousal:.6f}" if arousal is not None else "",
                    "dominance": f"{dominance:.6f}" if dominance is not None else "",
                }
            )

    # 결과 저장
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["ko", "key", "en", "valence", "arousal", "dominance"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n매칭 완료! 결과가 {OUTPUT_FILE}에 저장되었습니다.")
    print(
        f"총 {len(rows)}개 단어 중 {matched_count}개 매칭 성공, {len(failed_words)}개 실패"
    )

    if failed_words:
        print("\n매칭 실패한 단어 목록:")
        print("-" * 60)
        for ko_word, en_word in failed_words:
            print(f"  한국어: {ko_word:20s}  영어: {en_word}")
        print("-" * 60)
        print(f"\n총 {len(failed_words)}개 단어의 VAD 값을 찾지 못했습니다.")
        print("이 단어들은 수동으로 확인하거나 다른 번역을 시도해볼 수 있습니다.")
    else:
        print("\n모든 단어의 VAD 값이 성공적으로 매칭되었습니다!")


if __name__ == "__main__":
    main()
