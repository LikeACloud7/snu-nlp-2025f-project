#!/usr/bin/env python3
"""
데이터 포맷 변환 스크립트

raw 데이터를 README.md에 제시된 JSON 포맷으로 변환합니다.
"""

import json
import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, Optional


def load_emotions_mapping(emotions_tsv_path: str) -> Dict[str, Dict[str, any]]:
    """
    emotions.tsv 파일을 읽어서 key -> (en, valence, arousal, dominance) 매핑을 생성합니다.

    Args:
        emotions_tsv_path: emotions.tsv 파일 경로

    Returns:
        key를 키로 하고, en, valence, arousal, dominance를 값으로 하는 딕셔너리
    """
    emotion_map = {}

    with open(emotions_tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = row["key"]
            emotion_map[key] = {
                "en": row["en"],
                "valence": float(row["valence"]),
                "arousal": float(row["arousal"]),
                "dominance": float(row["dominance"]),
            }

    return emotion_map


def extract_turn_number(key: str) -> int:
    """
    HS01, SS01 같은 키에서 숫자 부분을 추출합니다.

    Args:
        key: "HS01", "SS02" 같은 형식의 키

    Returns:
        추출된 숫자 (예: 1, 2)
    """
    match = re.search(r"(\d+)", key)
    return int(match.group(1)) if match else 0


def convert_record(
    record: dict, emotion_map: Dict[str, Dict[str, any]]
) -> Optional[dict]:
    """
    단일 레코드를 변환합니다.

    Args:
        record: 입력 레코드
        emotion_map: 감정 매핑 딕셔너리

    Returns:
        변환된 레코드 또는 None (변환 실패 시)
    """
    try:
        # 감정 키 추출
        emotion_key = record["profile"]["emotion"]["type"]

        # 감정 정보 조회
        if emotion_key not in emotion_map:
            print(f"Warning: Emotion key {emotion_key} not found in emotions.tsv")
            return None

        emotion_info = emotion_map[emotion_key]

        # 대화 턴 추출 및 변환
        content = record["talk"]["content"]
        turns = []

        # 키를 숫자 순서로 정렬
        sorted_keys = sorted(content.keys(), key=extract_turn_number)

        for key in sorted_keys:
            text = content[key].strip()

            # 빈 텍스트는 무시
            if not text:
                continue

            # HS** -> client, SS** -> therapist
            if key.startswith("HS"):
                speaker = "client"
            elif key.startswith("SS"):
                speaker = "therapist"
            else:
                # 알 수 없는 형식은 스킵
                continue

            turns.append({"speaker": speaker, "text": text})

        # 턴이 없으면 None 반환
        if not turns:
            return None

        # 최종 레코드 구성
        converted_record = {
            "turns": turns,
            "emotion": emotion_info["en"],
            "vad": {
                "valence": emotion_info["valence"],
                "arousal": emotion_info["arousal"],
                "dominance": emotion_info["dominance"],
            },
        }

        return converted_record

    except KeyError as e:
        print(f"Error: Missing key in record: {e}")
        return None
    except Exception as e:
        print(f"Error converting record: {e}")
        return None


def format_data(
    input_path: str, output_path: str, emotions_tsv_path: str, is_sample: bool = False
):
    """
    데이터를 변환하여 저장합니다.

    Args:
        input_path: 입력 JSON 파일 경로
        output_path: 출력 JSONL 파일 경로
        emotions_tsv_path: emotions.tsv 파일 경로
        is_sample: True이면 첫 번째 데이터만 변환
    """
    # 감정 매핑 로드
    print(f"Loading emotions mapping from {emotions_tsv_path}...")
    emotion_map = load_emotions_mapping(emotions_tsv_path)
    print(f"Loaded {len(emotion_map)} emotion mappings")

    # 입력 파일 읽기
    print(f"Reading input file: {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input file must contain a JSON array")

    print(f"Found {len(data)} records")

    # 샘플 모드인 경우 첫 번째만 처리
    if is_sample:
        data = data[:1]
        print("Sample mode: processing only the first record")

    # 출력 디렉토리 생성
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 변환 및 저장
    print(f"Converting and writing to {output_path}...")
    converted_count = 0
    skipped_count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for i, record in enumerate(data):
            converted = convert_record(record, emotion_map)

            if converted is None:
                skipped_count += 1
                continue

            # JSONL 형식으로 저장 (한 줄에 하나의 JSON 객체)
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            converted_count += 1

            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(data)} records...")

    print("\nConversion complete!")
    print(f"  Converted: {converted_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Output file: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw data to formatted JSON format"
    )
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument(
        "--output", type=str, required=True, help="Output JSONL file path"
    )
    parser.add_argument(
        "--emotions-tsv",
        type=str,
        default="data/raw/metadata/emotions.tsv",
        help="Path to emotions.tsv file (default: data/raw/metadata/emotions.tsv)",
    )
    parser.add_argument(
        "--isSample",
        action="store_true",
        help="Convert only the first record (for testing)",
    )

    args = parser.parse_args()

    # 상대 경로를 절대 경로로 변환
    script_dir = Path(__file__).parent.parent.parent
    input_path = (
        args.input
        if os.path.isabs(args.input)
        else os.path.join(script_dir, args.input)
    )
    output_path = (
        args.output
        if os.path.isabs(args.output)
        else os.path.join(script_dir, args.output)
    )
    emotions_tsv_path = (
        args.emotions_tsv
        if os.path.isabs(args.emotions_tsv)
        else os.path.join(script_dir, args.emotions_tsv)
    )

    format_data(input_path, output_path, emotions_tsv_path, args.isSample)


if __name__ == "__main__":
    main()
