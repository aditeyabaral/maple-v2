import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AdamW

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

dataset = load_dataset("glue", "cola", split="train")
dataset = dataset.map(
    lambda examples: tokenizer(examples["sentence"],
                               truncation=True,
                               padding="max_length"), batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
dataset = torch.utils.data.DataLoader(dataset, batch_size=4)

epochs = 10
optimizer = AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5)

auth_token = ""
hub_model_name = ""
hub_organization_or_username = ""

for epoch in tqdm(range(epochs)):
    model.train()
    total_epoch_loss = 0
    num_batches = 0
    for batch in tqdm(dataset):
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["label"].to(device)
        )
        loss = outputs.loss
        total_epoch_loss += loss
        loss.backward()
        optimizer.step()
    total_epoch_loss /= num_batches
    scheduler.step(total_epoch_loss)
    print(f"Epoch {epoch} loss: {total_epoch_loss}")
    model.push_to_hub(f"{hub_organization_or_username}/{hub_model_name}", use_auth_token=auth_token)
    tokenizer.push_to_hub(f"{hub_organization_or_username}/{hub_model_name}", use_auth_token=auth_token)
