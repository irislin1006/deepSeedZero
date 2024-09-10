from datetime import datetime
import numpy as np

from datasets import load_dataset, load_metric
from data_utils import change_data

from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import torch


def train(args):
    model_checkpoint = args.model_checkpoint
    print(f"model_checkpoint:  {model_checkpoint} @@@###")
    Targs = TrainingArguments(
        report_to="tensorboard",
        output_dir=args.output_dir,
        evaluation_strategy=args.eval_save_strategy,
        save_strategy=args.eval_save_strategy,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=False,
        overwrite_output_dir=True,
        deepspeed=args.deepspeed,
        bf16=args.bf16,
    )

    # Load the dataset


    ds = load_dataset("json", data_files={"train": args.train_dataset_path, "test": args.test_dataset_path})
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    # Define a padding token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # or set a specific padding token
    ds = change_data(ds, args.data_type)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # def model_init():
    #     return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, return_dict=True)
    def model_init():
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, return_dict=True)
        model.config.pad_token_id = tokenizer.pad_token_id  # Set the pad_token_id in the model configuration
        print(f"model.config device: {model.config.device}")
        model.to(device)
        return model

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=409, padding="max_length", return_tensors="pt")

    encoded_ds = ds.map(preprocess_function, batched=True, num_proc=100)
    encoded_ds = encoded_ds.rename_column('label', 'labels')
    encoded_ds.set_format(type='torch', columns=['text', 'input_ids', 'attention_mask', 'labels'])
    print(f"First train sample device: {next(iter(encoded_ds['train']))['input_ids'].device}")
    print(f"First test sample device: {next(iter(encoded_ds['test']))['input_ids'].device}")

    # encoded_ds.set_format("torch", device="cuda") 
    num_labels = {0: 108, 1: 48, 2: 27, 3: 12}
    current_time = datetime.now().strftime("%m%d")
    args.output_dir = f"{args.output_dir}/{current_time}_{args.sub_name}"
    if not args.use_param_search:
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels[args.data_type])
        model.config.pad_token_id = tokenizer.pad_token_id  # Set the pad_token_id in the model configuration
        model.to(device)

    def compute_metrics(p):
        metric = load_metric("accuracy")
        preds = np.argmax(p.predictions, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    if args.use_param_search:
        trainer = Trainer(
            args=Targs,
            train_dataset=encoded_ds["train"],
            eval_dataset=encoded_ds["test"],
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
        )
        best_run = trainer.hyperparameter_search(n_trials=4, direction="maximize")
        for n, v in best_run.hyperparameters.items():
            setattr(trainer.args, n, v)
        trainer.train()
    else:
        trainer = Trainer(
            model=model,
            args=Targs,
            train_dataset=encoded_ds["train"],
            eval_dataset=encoded_ds["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        trainer.train()
