import argparse
import json
import os
from inspect import signature

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from utils import EmotionCatalog, ensure_catalog, load_prompt_templates, prepare_sft_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA SFT for VAD value generation.")
    parser.add_argument(
        "--dataset_name",
        help="Hugging Face dataset identifier or dataset directory to load via load_dataset().",
    )
    parser.add_argument(
        "--train_file",
        help="Local JSON/JSONL file to use for the training split (loaded via datasets.load_dataset('json', ...)).",
    )
    parser.add_argument(
        "--validation_file",
        help="Optional JSON/JSONL file for evaluation split when --eval_split is provided.",
    )
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="validation")
    parser.add_argument("--vad_key", default="vad")
    parser.add_argument("--output_dir", default="outputs/vad-gen")
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--prompt_config", required=True)
    parser.add_argument(
        "--target_speaker",
        default="client",
        help="Speaker id to estimate VAD for (e.g., client, therapist, last). Use 'last' to select the final turn.",
    )
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def prepare_dataset(dataset, template, tokenizer, vad_key, target_speaker):
    if dataset is None:
        return None

    def target_fn(example):
        payload = example[vad_key]
        response = {
            "valence": round(float(payload["valence"]), 3),
            "arousal": round(float(payload["arousal"]), 3),
            "dominance": round(float(payload["dominance"]), 3),
        }
        return json.dumps(response, ensure_ascii=False) + "\n"

    return prepare_sft_dataset(
        dataset,
        tokenizer,
        template,
        target_fn,
        {},
        target_speaker=target_speaker,
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.dataset_name and not args.train_file:
        raise ValueError("Provide either --dataset_name or --train_file.")

    if args.train_file:
        data_files = {args.train_split: args.train_file}
        if args.validation_file and args.eval_split:
            data_files[args.eval_split] = args.validation_file
        dataset = load_dataset("json", data_files=data_files)
    else:
        dataset = load_dataset(args.dataset_name)

    train_ds = dataset[args.train_split]
    eval_ds = dataset[args.eval_split] if args.eval_split and args.eval_split in dataset else None

    catalog = EmotionCatalog.from_dataset(train_ds)
    ensure_catalog(args.output_dir, catalog)

    prompts = load_prompt_templates(args.prompt_config)
    template = prompts["vad_generation"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_sft = prepare_dataset(train_ds, template, tokenizer, args.vad_key, args.target_speaker)
    eval_sft = prepare_dataset(eval_ds, template, tokenizer, args.vad_key, args.target_speaker)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_steps=500,
        logging_steps=50,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        bf16=True,
        save_strategy="epoch",
        report_to="none",
        seed=args.seed,
        max_grad_norm=1.0,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )
    eval_strategy_value = "steps" if eval_sft else "no"
    training_sig = signature(SFTConfig.__init__)
    if "evaluation_strategy" in training_sig.parameters:
        training_kwargs["evaluation_strategy"] = eval_strategy_value
    else:
        training_kwargs["eval_strategy"] = eval_strategy_value

    training_args = SFTConfig(**training_kwargs)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_sft,
        eval_dataset=eval_sft,
        args=training_args,
    )
    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
