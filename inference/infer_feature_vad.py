import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoTokenizer, LlamaForSequenceClassification

from utils import EmotionCatalog, build_chat_prompt, extract_context_target, load_prompt_templates


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for LoRA feature-based VAD regression.")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--conversation_path", required=True)
    parser.add_argument("--prompt_config", required=True)
    parser.add_argument(
        "--target_speaker",
        default="client",
        help="Speaker id to regress VAD for (e.g., client, therapist, last). Use 'last' to take the final turn.",
    )
    return parser.parse_args()


def load_turns(path: str):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["turns"]


def main():
    args = parse_args()
    turns = load_turns(args.conversation_path)

    catalog = EmotionCatalog.from_file(f"{args.adapter_dir}/emotion_catalog.json")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = load_prompt_templates(args.prompt_config)
    prompt_kwargs = extract_context_target(turns, args.target_speaker)
    prompt = build_chat_prompt(
        tokenizer,
        prompts["feature_extraction"],
        add_generation_prompt=False,
        **prompt_kwargs,
    )

    base_model = LlamaForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        device_map="auto",
        torch_dtype="auto",
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.problem_type = "regression"
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(0)
    vad = logits.cpu().tolist()
    payload = {"valence": vad[0], "arousal": vad[1], "dominance": vad[2]}
    mapped = catalog.map_vad_to_label(vad)
    print(json.dumps({"vad": payload, "mapped_emotion": mapped}, ensure_ascii=False))


if __name__ == "__main__":
    main()
