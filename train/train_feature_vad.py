import argparse
import os

import evaluate
import numpy as np
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, DataCollatorWithPadding, LlamaForSequenceClassification, Trainer, TrainingArguments

from utils import (EmotionCatalog, build_chat_prompt, ensure_catalog, extract_context_target,
                   load_prompt_templates)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA feature-based VAD regression.")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="validation")
    parser.add_argument("--vad_key", default="vad")
    parser.add_argument("--output_dir", default="outputs/feature-vad")
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--prompt_config", required=True)
    parser.add_argument(
        "--target_speaker",
        default="client",
        help="Speaker id to regress VAD for (e.g., client, therapist, last). Use 'last' to take the final turn.",
    )
    parser.add_argument("--max_seq_length", type=int, default=1536)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    return parser.parse_args()


def tokenize_dataset(dataset, tokenizer, template, max_seq_length, vad_key, target_speaker):
    if dataset is None:
        return None

    def _tokenize(example):
        info = extract_context_target(example["turns"], target_speaker)
        prompt = build_chat_prompt(
            tokenizer,
            template,
            add_generation_prompt=False,
            **info,
        )
        encoded = tokenizer(prompt, truncation=True, max_length=max_seq_length)
        vad = example[vad_key]
        encoded["labels"] = [
            float(vad["valence"]),
            float(vad["arousal"]),
            float(vad["dominance"]),
        ]
        return encoded

    tokenized = dataset.map(_tokenize, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch")
    return tokenized


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_dataset(args.dataset_name)
    train_ds = dataset[args.train_split]
    eval_ds = dataset[args.eval_split] if args.eval_split in dataset else None

    catalog = EmotionCatalog.from_dataset(train_ds)
    ensure_catalog(args.output_dir, catalog)

    prompts = load_prompt_templates(args.prompt_config)
    template = prompts["feature_extraction"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_tok = tokenize_dataset(
        train_ds,
        tokenizer,
        template,
        args.max_seq_length,
        args.vad_key,
        args.target_speaker,
    )
    eval_tok = tokenize_dataset(
        eval_ds,
        tokenizer,
        template,
        args.max_seq_length,
        args.vad_key,
        args.target_speaker,
    )

    model = LlamaForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        device_map="auto",
        torch_dtype="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.problem_type = "regression"
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)

    mae = evaluate.load("mae")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        mae_value = mae.compute(predictions=preds, references=labels)["mae"]
        rmse = np.sqrt(np.mean((preds - labels) ** 2))
        return {"mae": mae_value, "rmse": rmse}

    collator = DataCollatorWithPadding(tokenizer, padding=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="steps" if eval_tok else "no",
        eval_steps=500,
        logging_steps=50,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        bf16=True,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics if eval_tok else None,
    )
    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
