from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import yaml

SPEAKER_MAP = {
    "counselor": "Counselor",
    "therapist": "Therapist",
    "client": "Client",
    "patient": "Patient",
    "user": "User",
    "assistant": "Assistant",
}

_PROMPT_CACHE: MutableMapping[str, Mapping[str, "PromptTemplate"]] = {}


@dataclass(frozen=True)
class PromptTemplate:
    system: str
    user: str


def load_prompt_templates(path: str) -> Mapping[str, PromptTemplate]:
    cache_key = str(Path(path).resolve())
    if cache_key not in _PROMPT_CACHE:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        prompts: Dict[str, PromptTemplate] = {}
        for key, section in raw.items():
            if not isinstance(section, Mapping):
                raise ValueError(f"Prompt section '{key}' must be a mapping.")
            system = section.get("system")
            user = section.get("user")
            if system is None or user is None:
                raise ValueError(f"Prompt section '{key}' must contain 'system' and 'user'.")
            prompts[key] = PromptTemplate(system=system, user=user)
        _PROMPT_CACHE[cache_key] = prompts
    return _PROMPT_CACHE[cache_key]


def build_chat_prompt(
    tokenizer,
    template: PromptTemplate,
    *,
    add_generation_prompt: bool,
    **kwargs: Any,
) -> str:
    system_text = template.system.format(**kwargs).strip()
    user_text = template.user.format(**kwargs).strip()

    messages: List[Dict[str, str]] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    if user_text:
        messages.append({"role": "user", "content": user_text})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def format_dialogue(turns: Sequence[Mapping[str, str]]) -> str:
    lines: List[str] = []
    for turn in turns:
        speaker = SPEAKER_MAP.get(turn.get("speaker", "").lower(), turn.get("speaker", "Speaker").title())
        text = turn.get("text", "").strip().replace("\n", " ")
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def extract_context_target(
    turns: Sequence[Mapping[str, str]],
    target_speaker: Optional[str] = None,
) -> Dict[str, str]:
    if not turns:
        return {
            "context": "없음",
            "target_utterance": "정보 없음",
            "target_utterance_text": "",
            "target_speaker": "Unknown",
            "full_conversation": "",
        }

    normalized_target = None
    if target_speaker:
        lowered = target_speaker.strip().lower()
        if lowered not in {"", "last", "auto", "any"}:
            normalized_target = lowered

    entries: List[Dict[str, Any]] = []
    target_index = None
    for idx, turn in enumerate(turns):
        raw_speaker = (turn.get("speaker") or "").strip()
        speaker_norm = raw_speaker.lower()
        speaker_label = SPEAKER_MAP.get(speaker_norm, raw_speaker.title() if raw_speaker else "Speaker")
        text = (turn.get("text") or "").strip().replace("\n", " ")
        line = f"{speaker_label}: {text}"
        entries.append(
            {
                "index": idx,
                "speaker": speaker_label,
                "speaker_norm": speaker_norm,
                "text": text,
                "line": line,
            }
        )
        if normalized_target is None:
            target_index = idx
        elif speaker_norm == normalized_target:
            target_index = idx

    if target_index is None:
        target_index = len(entries) - 1

    target_entry = entries[target_index]
    context_lines = [entry["line"] for entry in entries if entry["index"] < target_index]
    context_text = "\n".join(context_lines).strip()
    if not context_text:
        context_text = "없음"

    full_conversation = "\n".join(entry["line"] for entry in entries)

    return {
        "context": context_text,
        "target_utterance": f"{target_entry['speaker']}: {target_entry['text']}",
        "target_utterance_text": target_entry["text"],
        "target_speaker": target_entry["speaker"],
        "full_conversation": full_conversation,
    }


def label_block(labels: Iterable[str]) -> str:
    return "\n".join(f"- {label}" for label in labels)


@dataclass
class EmotionCatalog:
    labels: List[str]
    label2id: Dict[str, int]
    id2label: Dict[int, str]
    vad_prototypes: Dict[str, List[float]]

    @classmethod
    def from_dataset(cls, dataset, label_key: str = "emotion", vad_key: str = "vad") -> "EmotionCatalog":
        labels = sorted(set(dataset[label_key]))
        label2id = {label: idx for idx, label in enumerate(labels)}
        id2label = {idx: label for label, idx in label2id.items()}

        vad_prototypes: Dict[str, List[float]] = {label: [0.0, 0.0, 0.0] for label in labels}
        counts: Dict[str, int] = {label: 0 for label in labels}
        if vad_key in dataset.column_names:
            for example in dataset:
                vad = example.get(vad_key)
                if vad is None:
                    continue
                vec = np.array(
                    [float(vad["valence"]), float(vad["arousal"]), float(vad["dominance"])],
                    dtype=np.float32,
                )
                label = example[label_key]
                vad_prototypes[label] = list(np.array(vad_prototypes[label]) + vec)
                counts[label] += 1
            for label in labels:
                if counts[label]:
                    vad_prototypes[label] = list(np.array(vad_prototypes[label]) / counts[label])
                else:
                    vad_prototypes[label] = [0.5, 0.5, 0.5]
        return cls(labels, label2id, id2label, vad_prototypes)

    @classmethod
    def from_file(cls, path: str) -> "EmotionCatalog":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        labels = payload["labels"]
        label2id = {label: int(idx) for label, idx in payload["label2id"].items()}
        id2label = {idx: label for label, idx in label2id.items()}
        vad_proto = {label: [float(x) for x in values] for label, values in payload.get("vad_prototypes", {}).items()}
        if not vad_proto:
            vad_proto = {label: [0.5, 0.5, 0.5] for label in labels}
        return cls(labels, label2id, id2label, vad_proto)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "labels": self.labels,
            "label2id": self.label2id,
            "vad_prototypes": self.vad_prototypes,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def map_vad_to_label(self, vad_vector: Sequence[float]) -> str:
        target = np.array(vad_vector, dtype=np.float32)
        best_label, best_dist = None, float("inf")
        for label, proto in self.vad_prototypes.items():
            proto_vec = np.array(proto, dtype=np.float32)
            dist = np.linalg.norm(target - proto_vec)
            if dist < best_dist:
                best_dist = dist
                best_label = label
        return best_label or self.labels[0]


def prepare_sft_dataset(
    dataset,
    tokenizer,
    template: PromptTemplate,
    target_fn,
    static_kwargs: Mapping[str, Any],
    target_speaker: Optional[str] = None,
) -> Any:
    if dataset is None:
        return None
    remove_columns = dataset.column_names

    def _format(example):
        kwargs = dict(static_kwargs)
        turn_info = extract_context_target(example["turns"], target_speaker)
        kwargs.update(turn_info)
        prompt = build_chat_prompt(
            tokenizer,
            template,
            add_generation_prompt=True,
            **kwargs,
        )
        example["text"] = prompt + target_fn(example)
        return example

    return dataset.map(_format, remove_columns=remove_columns, desc="Preparing SFT dataset")


def ensure_catalog(output_dir: str, catalog: EmotionCatalog) -> str:
    output_path = Path(output_dir) / "emotion_catalog.json"
    catalog.save(str(output_path))
    return str(output_path)
