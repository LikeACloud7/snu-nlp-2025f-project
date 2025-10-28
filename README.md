# snu-nlp-2025f-project

LoRA fine-tuning experiments for counselling-dialog emotion understanding.  
Four modelling strategies are implemented on multi-turn counsellor–client conversations:

1. **Generative Emotion Classification** – prompt the model to output a single label directly.  
2. **Generative VAD Estimation** – predict Valence/Arousal/Dominance scores and map them onto labels.  
3. **Feature-Based Classification** – encode the dialogue with a feature prompt and classify via a sequence-classification head.  
4. **Feature-Based VAD Regression** – regress VAD scores from the feature prompt, then map to labels.

Meta-Llama-3.1-8B-Instruct is used as the base model for every configuration, and all adapters are trained with LoRA.

---

## Environment

- **Python**: 3.10 (recommended)  
- **Dependencies**: install via `pip install -r requirements.txt`

Main libraries: `transformers`, `accelerate`, `trl`, `peft`, `datasets`, `evaluate`, `scikit-learn`, `numpy`, `pyyaml`.

---

## Dataset Expectations

All scripts assume a Hugging Face datasets-compatible structure (JSON/Parquet/etc.) with fields:

```jsonc
{
  "turns": [
    {"speaker": "client", "text": "..."},
    {"speaker": "therapist", "text": "..."}
  ],
  "emotion": "Gratitude",             // String label
  "vad": {                            // Optional for label-only runs, required for VAD tasks
    "valence": 0.74,
    "arousal": 0.53,
    "dominance": 0.61
  }
}
```

Create `train`, `validation`, and (optionally) `test` splits or update the script arguments to match your custom split names.
For emotion/VAD supervision, the target label must correspond to the final occurrence of the chosen speaker (default: the `client`). Only the turns *before* that utterance appear in `{context}`; later turns are ignored.

---

## Prompt Configuration

Prompts are centralised in `config/prompts.yaml`. Each entry has `system`/`user` sections written in Korean, rendered with the official Llama chat template. The user message separates the **dialogue context** (`{context}`) and the **target utterance** (`{target_utterance}`) so the model focuses on the final speaker turn while still seeing the preceding conversation.

---

## Training

All training scripts live in `train/` and accept Hugging Face dataset identifiers or local dataset paths.

### 1. Direct Emotion Classification (Generative)
```bash
python train/train_lora_direct_classification.py \
  --dataset_name path/to/dataset \
  --train_split train \
  --eval_split validation \
  --label_key emotion \
  --target_speaker client \
  --prompt_config config/prompts.yaml \
  --output_dir outputs/direct-class
```

### 2. VAD Generation (Generative)
```bash
python train/train_lora_vad_generation.py \
  --dataset_name path/to/dataset \
  --vad_key vad \
  --target_speaker client \
  --prompt_config config/prompts.yaml \
  --output_dir outputs/vad-gen
```

### 3. Feature-Based Classification (Sequence Classification Head)
```bash
python train/train_feature_classifier.py \
  --dataset_name path/to/dataset \
  --label_key emotion \
  --target_speaker client \
  --prompt_config config/prompts.yaml \
  --output_dir outputs/feature-class
```

### 4. Feature-Based VAD Regression
```bash
python train/train_feature_vad.py \
  --dataset_name path/to/dataset \
  --vad_key vad \
  --target_speaker client \
  --prompt_config config/prompts.yaml \
  --output_dir outputs/feature-vad
```

---

## Inference

Inference scripts reside in `inference/`. Supply the path to the trained adapter, base model name, prompt configuration, and a JSON conversation file matching the dataset schema.

### Direct Emotion Prediction
```bash
python inference/infer_lora_direct_classification.py \
  --adapter_dir outputs/direct-class \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --target_speaker client \
  --prompt_config config/prompts.yaml \
  --conversation_path sample_dialogue.json
```

### VAD Generation → Label Mapping
```bash
python inference/infer_lora_vad_generation.py \
  --adapter_dir outputs/vad-gen \
  --target_speaker client \
  --prompt_config config/prompts.yaml \
  --conversation_path sample_dialogue.json
```

### Feature-Based Classification
```bash
python inference/infer_feature_classifier.py \
  --adapter_dir outputs/feature-class \
  --target_speaker client \
  --prompt_config config/prompts.yaml \
  --conversation_path sample_dialogue.json
```

### Feature-Based VAD Regression
```bash
python inference/infer_feature_vad.py \
  --adapter_dir outputs/feature-vad \
  --target_speaker client \
  --prompt_config config/prompts.yaml \
  --conversation_path sample_dialogue.json
```

- All evaluations focus on classification metrics; compute macro/micro F1 and ROC-AUC for the target emotion labels.

---

## Repository Layout

```
config/            # Prompt configuration (YAML)
train/             # Training scripts for the four modes
inference/         # Matching inference scripts
requirements.txt   # Python dependencies
utils.py           # Shared utilities: prompt rendering, dialogue formatting, catalog helpers
```

---
