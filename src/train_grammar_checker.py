import argparse

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer

parser = argparse.ArgumentParser("Train Grammar Checker on GLUE CoLA")
parser.add_argument("--model-name", type=str, default="jxuhf/roberta-base-finetuned-cola")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--learning-rate", type=float, default=1e-8)
parser.add_argument("--hub-username-or-org", type=str, default=None)
parser.add_argument("--hub-repo-name", type=str, default=None)
parser.add_argument("--hub-auth-token", type=str, default=None)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(args.device)

dataset = load_dataset("glue", "cola", split="train")
dataset = dataset.map(lambda examples: tokenizer(examples["sentence"], truncation=True, padding="max_length"),
                      batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
dataset = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

epochs = args.epochs
optimizer = AdamW(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5)

for epoch in tqdm(range(epochs)):
    model.train()
    total_epoch_loss = 0
    num_batches = 0
    for batch in tqdm(dataset):
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"].to(args.device),
            attention_mask=batch["attention_mask"].to(args.device),
            labels=batch["label"].to(args.device)
        )
        loss = outputs.loss
        total_epoch_loss += loss
        loss.backward()
        optimizer.step()
    total_epoch_loss /= num_batches
    scheduler.step(total_epoch_loss)
    print(f"Epoch {epoch} loss: {total_epoch_loss}")
    if args.auth_token and args.hub_username_or_org and args.hub_repo_name:
        model.push_to_hub(
            f"{args.hub_username_or_org}/{args.hub_repo_name}",
            commit_message=f"Epoch {epoch} loss: {total_epoch_loss}",
            use_auth_token=args.auth_token
        )
        tokenizer.push_to_hub(
            f"{args.hub_username_or_org}/{args.hub_repo_name}",
            use_auth_token=args.auth_token
        )
