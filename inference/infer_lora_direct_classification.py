import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (EmotionCatalog, build_chat_prompt, extract_context_target, label_block,
                   load_prompt_templates)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for LoRA direct emotion classification.")
    parser.add_argument(
        "--adapter_dir",
        help="LoRA adapter directory. If omitted, the base model is used without adapters.",
    )
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--conversation_path",
        required=True,
        help="Path to a JSON/JSONL file. JSONL will run inference for every record.",
    )
    parser.add_argument("--prompt_config", required=True)
    parser.add_argument(
        "--catalog_path",
        help="Path to emotion_catalog.json when running without an adapter.",
    )
    parser.add_argument(
        "--output_path",
        help="Optional file to save the aggregated inference outputs as JSON.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--target_speaker",
        default="client",
        help="Speaker id to classify (e.g., client, therapist, last). Use 'last' to select the final turn.",
    )
    return parser.parse_args()


def load_conversations(path: str):
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


def main():
    args = parse_args()
    conversations = load_conversations(args.conversation_path)
    if not conversations:
        raise ValueError("No conversations found in the provided file.")

    catalog_path = None
    if args.adapter_dir:
        catalog_path = f"{args.adapter_dir}/emotion_catalog.json"
    elif args.catalog_path:
        catalog_path = args.catalog_path
    if catalog_path is None:
        raise ValueError("Provide --adapter_dir or --catalog_path to load the emotion catalog.")

    catalog = EmotionCatalog.from_file(catalog_path)
    tokenizer_source = args.adapter_dir or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = load_prompt_templates(args.prompt_config)
    template = prompts["classification"]

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    if args.adapter_dir:
        model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    else:
        model = base_model
    model.eval()

    results = []
    iterator = tqdm(conversations, desc="Running inference", unit="sample")
    for idx, entry in enumerate(iterator):
        prompt_kwargs = extract_context_target(entry["turns"], args.target_speaker)
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
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.temperature > 0,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
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

    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    elif len(results) == 1:
        print(json.dumps(results[0], ensure_ascii=False))
    else:
        for record in results:
            print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
