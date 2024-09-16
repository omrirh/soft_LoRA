import torch
import numpy as np
from typing import Tuple, Any, Dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import evaluate


def load_glue_data(task_name: str = "sst2"):
    dataset = load_dataset("glue", task_name)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "label"])

    return tokenized_datasets


def initialize_lora_model() -> torch.nn.Module:
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.05
    )

    peft_model = get_peft_model(model, lora_config)

    return peft_model


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train_model(model: torch.nn.Module, tokenized_datasets) -> Tuple[Trainer, torch.nn.Module]:
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=5e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        seed=42
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics
    )

    trainer.train()

    return trainer, model


def main():
    tokenized_datasets = load_glue_data("sst2")
    model = initialize_lora_model()

    trainer, trained_model = train_model(model, tokenized_datasets)
    eval_results = trainer.evaluate()

    print(f"Evaluation results: {eval_results}")


if __name__ == "__main__":
    main()
