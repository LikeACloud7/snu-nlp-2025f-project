import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    EmotionCatalog,
    build_chat_prompt,
    extract_context_target,
    label_block,
    load_prompt_templates,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run validation/test inference for base and LoRA-tuned models."
    )
    parser.add_argument(
        "--conversation_path",
        default="data/validation.jsonl",
        help="Validation/Test JSON/JSONL file that will be used for inference.",
    )
    parser.add_argument(
        "--prompt_config",
        default="config/prompts.yaml",
        help="Path to the prompt template YAML file.",
    )
    parser.add_argument(
        "--model_name",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Identifier or local path for the original base model.",
    )
    parser.add_argument(
        "--classification_adapter",
        help="Optional directory that stores the direct classification LoRA adapter.",
    )
    parser.add_argument(
        "--vad_adapter",
        help="Optional directory that stores the VAD generation LoRA adapter.",
    )
    parser.add_argument(
        "--catalog_path",
        help="Emotion catalog JSON path. If omitted, adapters (when provided) are used to locate it.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/validation_inference",
        help="Directory to store inference outputs.",
    )
    parser.add_argument(
        "--classification_max_new_tokens",
        type=int,
        default=8,
        help="Max new tokens for direct classification generation.",
    )
    parser.add_argument(
        "--vad_max_new_tokens",
        type=int,
        default=32,
        help="Max new tokens for VAD generation.",
    )
    parser.add_argument(
        "--classification_temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for direct classification.",
    )
    parser.add_argument(
        "--vad_temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for VAD generation.",
    )
    parser.add_argument(
        "--target_speaker",
        default="client",
        help="Dialogue speaker id to classify (e.g., client, therapist, last).",
    )
    return parser.parse_args()


def load_conversations(path: str) -> List[Dict]:
    entries = []
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    def _normalize(payload, idx):
        if "turns" not in payload:
            raise ValueError("Each record must contain a 'turns' key.")
        meta = dict(payload)
        turns = meta.pop("turns")
        entry_id = meta.pop("id", idx)
        entries.append({"id": entry_id, "turns": turns, "metadata": meta})

    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                _normalize(payload, idx)
    else:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            for idx, item in enumerate(payload):
                _normalize(item, idx)
        elif isinstance(payload, dict):
            _normalize(payload, 0)
        else:
            raise ValueError("Unsupported conversation file format.")
    return entries


def parse_json(text: str) -> Dict[str, Optional[float]]:
    default = {"valence": None, "arousal": None, "dominance": None}
    try:
        start = text.index("{")
        end = text.rfind("}") + 1
        parsed = json.loads(text[start:end])
        if not isinstance(parsed, dict):
            return default
        result = dict(default)
        for key in result:
            if key in parsed:
                result[key] = parsed[key]
        return result
    except Exception:
        return default


def build_catalog_from_conversations(conversations: Sequence[Dict]) -> EmotionCatalog:
    label_totals: Dict[str, List[float]] = {}
    label_counts: Dict[str, int] = {}
    label_set: List[str] = []

    for entry in conversations:
        metadata = entry.get("metadata") or {}
        label = metadata.get("emotion")
        if label is None:
            continue
        if label not in label_totals:
            label_totals[label] = [0.0, 0.0, 0.0]
            label_counts[label] = 0
            label_set.append(label)
        vad = metadata.get("vad")
        if isinstance(vad, dict):
            label_totals[label][0] += float(vad.get("valence", 0.0))
            label_totals[label][1] += float(vad.get("arousal", 0.0))
            label_totals[label][2] += float(vad.get("dominance", 0.0))
            label_counts[label] += 1

    if not label_set:
        raise ValueError("Unable to infer an emotion catalog from the dataset metadata.")

    labels = sorted(label_set)
    vad_prototypes: Dict[str, List[float]] = {}
    for label in labels:
        count = label_counts.get(label, 0)
        if count:
            totals = label_totals[label]
            vad_prototypes[label] = [totals[idx] / count for idx in range(3)]
        else:
            vad_prototypes[label] = [0.5, 0.5, 0.5]

    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return EmotionCatalog(labels, label2id, id2label, vad_prototypes)


def resolve_catalog(
    conversations: Sequence[Dict],
    explicit_path: Optional[str],
    adapter_dirs: Iterable[Optional[str]],
) -> EmotionCatalog:
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    for adapter_dir in adapter_dirs:
        if not adapter_dir:
            continue
        adapter_path = Path(adapter_dir) / "emotion_catalog.json"
        candidates.append(adapter_path)

    for candidate in candidates:
        if candidate.is_file():
            return EmotionCatalog.from_file(str(candidate))

    return build_catalog_from_conversations(conversations)


_MODEL_CACHE: Dict[str, AutoModelForCausalLM] = {}
_TOKENIZER_CACHE: Dict[str, AutoTokenizer] = {}


def load_model(model_name: str, adapter_dir: Optional[str]) -> AutoModelForCausalLM:
    key = adapter_dir or "__base__"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(base_model, adapter_dir)
    else:
        model = base_model
    model.eval()
    _MODEL_CACHE[key] = model
    return model


def load_tokenizer(source: str) -> AutoTokenizer:
    if source in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[source]
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _TOKENIZER_CACHE[source] = tokenizer
    return tokenizer


def compute_accuracy(records: Sequence[Dict], prediction_key: str) -> Optional[Dict[str, float]]:
    total = 0
    correct = 0
    for record in records:
        reference = record.get("reference_emotion")
        prediction = record.get(prediction_key)
        if reference is None or prediction is None:
            continue
        total += 1
        if prediction == reference:
            correct += 1
    if total == 0:
        return None
    accuracy = correct / total if total else None
    return {"total": total, "correct": correct, "accuracy": accuracy}


def run_direct_classification(
    conversations: Sequence[Dict],
    catalog: EmotionCatalog,
    template,
    model_name: str,
    adapter_dir: Optional[str],
    max_new_tokens: int,
    temperature: float,
    target_speaker: str,
) -> List[Dict]:
    model = load_model(model_name, adapter_dir)
    tokenizer_source = adapter_dir or model_name
    tokenizer = load_tokenizer(tokenizer_source)

    results: List[Dict] = []
    iterator = tqdm(
        conversations,
        desc=f"Direct classification ({Path(adapter_dir).name if adapter_dir else 'base'})",
        unit="sample",
    )
    for idx, entry in enumerate(iterator):
        prompt_kwargs = extract_context_target(entry["turns"], target_speaker)
        prompt_kwargs["label_block"] = label_block(catalog.labels)
        prompt_text = build_chat_prompt(
            tokenizer,
            template,
            add_generation_prompt=True,
            **prompt_kwargs,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        prediction = generated.split()[0] if generated else ""
        if prediction not in catalog.labels:
            prediction = catalog.map_vad_to_label([0.5, 0.5, 0.5])

        record = {
            "index": idx,
            "prediction": prediction,
            "raw_output": generated,
        }
        if entry["id"] is not None:
            record["id"] = entry["id"]
        reference = entry["metadata"].get("emotion")
        if reference is not None:
            record["reference_emotion"] = reference
        results.append(record)
    return results


def run_vad_generation(
    conversations: Sequence[Dict],
    catalog: EmotionCatalog,
    template,
    model_name: str,
    adapter_dir: Optional[str],
    max_new_tokens: int,
    temperature: float,
    target_speaker: str,
) -> List[Dict]:
    model = load_model(model_name, adapter_dir)
    tokenizer_source = adapter_dir or model_name
    tokenizer = load_tokenizer(tokenizer_source)

    results: List[Dict] = []
    iterator = tqdm(
        conversations,
        desc=f"VAD generation ({Path(adapter_dir).name if adapter_dir else 'base'})",
        unit="sample",
    )
    for idx, entry in enumerate(iterator):
        prompt_kwargs = extract_context_target(entry["turns"], target_speaker)
        prompt = build_chat_prompt(
            tokenizer,
            template,
            add_generation_prompt=True,
            **prompt_kwargs,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        vad = parse_json(generated)
        mapped = None
        if None not in vad.values():
            mapped = catalog.map_vad_to_label(
                [vad["valence"], vad["arousal"], vad["dominance"]]
            )
        record = {
            "index": idx,
            "vad": vad,
            "mapped_emotion": mapped,
            "raw_output": generated,
        }
        if entry["id"] is not None:
            record["id"] = entry["id"]
        reference = entry["metadata"].get("emotion")
        if reference is not None:
            record["reference_emotion"] = reference
        reference_vad = entry["metadata"].get("vad")
        if reference_vad is not None:
            record["reference_vad"] = reference_vad
        results.append(record)
    return results


def save_records(path: Path, records: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def variant_name(adapter_dir: Optional[str]) -> str:
    if not adapter_dir:
        return "base"
    return Path(adapter_dir).name or "adapter"


def main():
    args = parse_args()
    conversations = load_conversations(args.conversation_path)
    if not conversations:
        raise ValueError("No conversations found in the provided file.")

    prompts = load_prompt_templates(args.prompt_config)
    classification_template = prompts["classification"]
    vad_template = prompts["vad_generation"]

    catalog = resolve_catalog(
        conversations,
        args.catalog_path,
        adapter_dirs=[args.classification_adapter, args.vad_adapter],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: List[Dict[str, Optional[float]]] = []

    # Base direct classification
    base_class_records = run_direct_classification(
        conversations,
        catalog,
        classification_template,
        args.model_name,
        adapter_dir=None,
        max_new_tokens=args.classification_max_new_tokens,
        temperature=args.classification_temperature,
        target_speaker=args.target_speaker,
    )
    base_class_path = output_dir / f"direct_classification__{variant_name(None)}.json"
    save_records(base_class_path, base_class_records)
    summary.append(
        {
            "task": "direct_classification",
            "variant": variant_name(None),
            "output_path": str(base_class_path),
            **(compute_accuracy(base_class_records, "prediction") or {}),
        }
    )

    # LoRA direct classification
    if args.classification_adapter:
        lora_class_records = run_direct_classification(
            conversations,
            catalog,
            classification_template,
            args.model_name,
            adapter_dir=args.classification_adapter,
            max_new_tokens=args.classification_max_new_tokens,
            temperature=args.classification_temperature,
            target_speaker=args.target_speaker,
        )
        lora_class_path = output_dir / f"direct_classification__{variant_name(args.classification_adapter)}.json"
        save_records(lora_class_path, lora_class_records)
        summary.append(
            {
                "task": "direct_classification",
                "variant": variant_name(args.classification_adapter),
                "output_path": str(lora_class_path),
                **(compute_accuracy(lora_class_records, "prediction") or {}),
            }
        )

    # Base VAD generation
    base_vad_records = run_vad_generation(
        conversations,
        catalog,
        vad_template,
        args.model_name,
        adapter_dir=None,
        max_new_tokens=args.vad_max_new_tokens,
        temperature=args.vad_temperature,
        target_speaker=args.target_speaker,
    )
    base_vad_path = output_dir / f"vad_generation__{variant_name(None)}.json"
    save_records(base_vad_path, base_vad_records)
    summary.append(
        {
            "task": "vad_generation",
            "variant": variant_name(None),
            "output_path": str(base_vad_path),
            **(compute_accuracy(base_vad_records, "mapped_emotion") or {}),
        }
    )

    # LoRA VAD generation
    if args.vad_adapter:
        lora_vad_records = run_vad_generation(
            conversations,
            catalog,
            vad_template,
            args.model_name,
            adapter_dir=args.vad_adapter,
            max_new_tokens=args.vad_max_new_tokens,
            temperature=args.vad_temperature,
            target_speaker=args.target_speaker,
        )
        lora_vad_path = output_dir / f"vad_generation__{variant_name(args.vad_adapter)}.json"
        save_records(lora_vad_path, lora_vad_records)
        summary.append(
            {
                "task": "vad_generation",
                "variant": variant_name(args.vad_adapter),
                "output_path": str(lora_vad_path),
                **(compute_accuracy(lora_vad_records, "mapped_emotion") or {}),
            }
        )

    summary_path = output_dir / "summary.json"
    save_records(summary_path, summary)
    print(f"Inference complete. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
