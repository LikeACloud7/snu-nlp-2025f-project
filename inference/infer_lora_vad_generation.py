import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import EmotionCatalog, build_chat_prompt, extract_context_target, load_prompt_templates


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for LoRA VAD generation.")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--conversation_path", required=True)
    parser.add_argument("--prompt_config", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument(
        "--target_speaker",
        default="client",
        help="Speaker id to estimate VAD for (e.g., client, therapist, last). Use 'last' to select the final turn.",
    )
    return parser.parse_args()


def load_turns(path: str):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["turns"]


def parse_json(text: str):
    try:
        start = text.index("{")
        end = text.index("}", start) + 1
        return json.loads(text[start:end])
    except Exception:
        return {"valence": None, "arousal": None, "dominance": None}


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
        prompts["vad_generation"],
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

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.temperature > 0,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    vad = parse_json(generated)
    if None not in vad.values():
        mapped = catalog.map_vad_to_label([vad["valence"], vad["arousal"], vad["dominance"]])
    else:
        mapped = None
    print(json.dumps({"vad": vad, "mapped_emotion": mapped, "raw_output": generated}, ensure_ascii=False))


if __name__ == "__main__":
    main()
