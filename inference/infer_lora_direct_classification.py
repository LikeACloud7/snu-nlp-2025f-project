import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (EmotionCatalog, build_chat_prompt, extract_context_target, label_block,
                   load_prompt_templates)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for LoRA direct emotion classification.")
    parser.add_argument("--adapter_dir", required=True, help="LoRA adapter directory.")
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--conversation_path", required=True, help="JSON file with `turns` list.")
    parser.add_argument("--prompt_config", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--target_speaker",
        default="client",
        help="Speaker id to classify (e.g., client, therapist, last). Use 'last' to select the final turn.",
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
    prompt_kwargs["label_block"] = label_block(catalog.labels)
    prompt_text = build_chat_prompt(
        tokenizer,
        prompts["classification"],
        add_generation_prompt=True,
        **prompt_kwargs,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

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
    prediction = generated.split()[0]
    if prediction not in catalog.labels:
        prediction = catalog.map_vad_to_label([0.5, 0.5, 0.5])
    print(json.dumps({"prediction": prediction, "raw_output": generated}, ensure_ascii=False))


if __name__ == "__main__":
    main()
